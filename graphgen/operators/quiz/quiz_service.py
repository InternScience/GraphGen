import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm, init_storage
from graphgen.models import QuizGenerator
from graphgen.utils import logger, run_concurrent


class QuizService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        graph_backend: str = "kuzu",
        kv_backend: str = "rocksdb",
        quiz_samples: int = 1,
    ):
        super().__init__(working_dir=working_dir, kv_backend=kv_backend, op_name="quiz")
        self.quiz_samples = quiz_samples
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )
        # { _trace_id: { "description": str, "quizzes": List[Tuple[str, str]] } }
        self.generator = QuizGenerator(self.llm_client)

    async def _process_single_quiz(self, item: tuple) -> dict | None:
        # if quiz in quiz_storage exists already, directly get it
        desc, index = item

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
                "index": index,
                "description": desc,
                "quizzes": quizzes,
            }
        except Exception as e:
            logger.error("Error when quizzing description %s: %s", item, e)
            return None

    def process(self, batch: list) -> pd.DataFrame:
        """
        Get all nodes and edges and quiz their descriptions using QuizGenerator.
        """
        items = []

        for item in batch:
            input_id = item["_trace_id"]
            node = item.get("node")
            edge = item.get("edge")

            if node and node.get("description"):
                items.append((input_id, node["description"], node["entity_name"]))
            elif edge and edge.get("description"):
                edge_key = (edge["src_id"], edge["tgt_id"])
                items.append((input_id, edge["description"], edge_key))
        if not items:
            return pd.DataFrame()

        logger.info("Total descriptions to quiz: %d", len(items))
        results = run_concurrent(
            self._process_single_quiz,
            [(desc, orig_id) for (_, desc, orig_id) in items],
            desc=f"Quizzing batch of {len(items)} descriptions",
            unit="description",
        )

        final_results = []
        meta_update = {}

        for (input_id, _, _), quiz_data in zip(items, results):
            if quiz_data is None:
                continue
            quiz_data["_trace_id"] = self.generate_trace_id(quiz_data)
            final_results.append(quiz_data)
            meta_update[input_id] = [quiz_data["_trace_id"]]

        if final_results:
            self.store(final_results, meta_update)

        return pd.DataFrame(final_results)
