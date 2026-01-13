import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm, init_storage
from graphgen.models import QuizGenerator
from graphgen.utils import compute_dict_hash, logger, run_concurrent


class QuizService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        graph_backend: str = "kuzu",
        kv_backend: str = "rocksdb",
        quiz_samples: int = 1,
    ):
        super().__init__(working_dir=working_dir, op_name="quiz_service")
        self.quiz_samples = quiz_samples
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )
        # { _quiz_id: { "description": str, "quizzes": List[Tuple[str, str]] } }
        self.quiz_storage: BaseKVStorage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="quiz"
        )
        self.generator = QuizGenerator(self.llm_client)

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        data = batch.to_dict(orient="records")
        self.graph_storage.reload()
        return self.quiz(data)

    async def _process_single_quiz(self, item: tuple) -> dict | None:
        # if quiz in quiz_storage exists already, directly get it
        index, desc = item
        _quiz_id = compute_dict_hash({"index": index, "description": desc})

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

    def quiz(self, batch) -> pd.DataFrame:
        """
        Get all nodes and edges and quiz their descriptions using QuizGenerator.
        """
        items = []

        for item in batch:
            node_data = item.get("node", [])
            edge_data = item.get("edge", [])

            if node_data:
                node_id = node_data["entity_name"]
                desc = node_data["description"]
                items.append((node_id, desc))
            if edge_data:
                edge_key = (edge_data["src_id"], edge_data["tgt_id"])
                desc = edge_data["description"]
                items.append((edge_key, desc))

        logger.info("Total descriptions to quiz: %d", len(items))

        results = run_concurrent(
            self._process_single_quiz,
            items,
            desc=f"Quizzing batch of {len(items)} descriptions",
            unit="description",
        )
        valid_results = [res for res in results if res]

        for res in valid_results:
            self.quiz_storage.upsert(
                {
                    res["_quiz_id"]: {
                        "description": res["description"],
                        "quizzes": res["quizzes"],
                    }
                }
            )
        self.quiz_storage.index_done_callback()
        return pd.DataFrame(valid_results)
