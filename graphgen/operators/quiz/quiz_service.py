from collections.abc import Iterable

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm, init_storage
from graphgen.models import QuizGenerator
from graphgen.utils import compute_dict_hash, logger, run_concurrent


class QuizService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        quiz_samples: int = 1,
        concurrency_limit: int = 200,
    ):
        super().__init__(working_dir=working_dir, op_name="quiz_service")
        self.quiz_samples = quiz_samples
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend="networkx", working_dir=working_dir, namespace="graph"
        )
        # { _quiz_id: { "description": str, "quizzes": List[Tuple[str, str]] } }
        self.quiz_storage: BaseKVStorage = init_storage(
            backend="rocksdb", working_dir=working_dir, namespace="quiz"
        )
        self.generator = QuizGenerator(self.llm_client)
        self.concurrency_limit = concurrency_limit

    def process(self, batch: pd.DataFrame) -> Iterable[pd.DataFrame]:
        # this operator does not consume any batch data
        # but for compatibility we keep the interface
        _ = batch.to_dict(orient="records")
        self.graph_storage.reload()
        yield from self.quiz()

    async def _process_single_quiz(self, item: tuple) -> dict | None:
        # if quiz in quiz_storage exists already, directly get it
        index, desc = item
        _quiz_id = compute_dict_hash({"index": index, "description": desc})
        if self.quiz_storage.get_by_id(_quiz_id):
            return None

        tasks = []
        for i in range(self.quiz_samples):
            if i > 0:
                tasks.append((desc, "TEMPLATE", "yes"))
            tasks.append((desc, "ANTI_TEMPLATE", "no"))
        try:
            quizzes = []
            for d, template_type, gt in tasks:
                prompt = self.generator.build_prompt_for_description(d, template_type)
                new_description = await self.llm_client.generate_answer(
                    prompt, temperature=1
                )
                rephrased_text = self.generator.parse_rephrased_text(new_description)
                quizzes.append((rephrased_text, gt))
            return {
                "_quiz_id": _quiz_id,
                "description": desc,
                "index": index,
                "quizzes": quizzes,
            }
        except Exception as e:
            logger.error("Error when quizzing description %s: %s", item, e)
            return None

    def quiz(self) -> Iterable[pd.DataFrame]:
        """
        Get all nodes and edges and quiz their descriptions using QuizGenerator.
        """
        edges = self.graph_storage.get_all_edges()
        nodes = self.graph_storage.get_all_nodes()

        items = []

        for edge in edges:
            edge_data = edge[2]
            desc = edge_data["description"]
            items.append(((edge[0], edge[1]), desc))

        for node in nodes:
            node_data = node[1]
            desc = node_data["description"]
            items.append((node[0], desc))

        logger.info("Total descriptions to quiz: %d", len(items))

        for i in range(0, len(items), self.concurrency_limit):
            batch_items = items[i : i + self.concurrency_limit]
            batch_results = run_concurrent(
                self._process_single_quiz,
                batch_items,
                desc=f"Quizzing descriptions ({i} / {i + len(batch_items)})",
                unit="description",
            )

            final_results = []
            for new_result in batch_results:
                if new_result:
                    self.quiz_storage.upsert(
                        {
                            new_result["_quiz_id"]: {
                                "description": new_result["description"],
                                "quizzes": new_result["quizzes"],
                            }
                        }
                    )
                    final_results.append(new_result)
            self.quiz_storage.index_done_callback()
            yield pd.DataFrame(final_results)
