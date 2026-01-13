from typing import Any, Dict

import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator, QAPair
from graphgen.common import init_llm, init_storage
from graphgen.utils import logger, run_concurrent


class EvaluateService(BaseOperator):
    """
    1. KG Quality Evaluation
    2. QA Quality Evaluation
    """

    def __init__(
        self,
        working_dir: str = "cache",
        metrics: list[str] = None,
        graph_backend: str = "kuzu",
        kv_backend: str = "rocksdb",
        **kwargs,
    ):
        super().__init__(working_dir=working_dir, op_name="evaluate_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.metrics = metrics or []
        self.kwargs = kwargs
        self.graph_storage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )
        self.chunk_storage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="chunk"
        )

        # Initialize evaluators
        self.qa_evaluators = {}
        self.kg_evaluators = {}
        self._init_evaluators()

    def _init_evaluators(self):
        """Initialize QA and KG evaluators based on metrics."""
        for metric in self.metrics:
            if metric == "qa_length":
                from graphgen.models import LengthEvaluator

                self.qa_evaluators[metric] = LengthEvaluator()
            elif metric == "qa_mtld":
                from graphgen.models import MTLDEvaluator

                self.qa_evaluators[metric] = MTLDEvaluator(
                    **self.kwargs.get("mtld_params", {})
                )
            elif metric == "qa_reward_score":
                from graphgen.models import RewardEvaluator

                self.qa_evaluators[metric] = RewardEvaluator(
                    **self.kwargs.get("reward_params", {})
                )
            elif metric == "qa_uni_score":
                from graphgen.models import UniEvaluator

                self.qa_evaluators[metric] = UniEvaluator(
                    **self.kwargs.get("uni_params", {})
                )
            elif metric == "kg_accuracy":
                from graphgen.models import AccuracyEvaluator

                self.kg_evaluators[metric] = AccuracyEvaluator(
                    graph_storage=self.graph_storage,
                    chunk_storage=self.chunk_storage,
                    llm_client=self.llm_client,
                )
            elif metric == "kg_consistency":
                from graphgen.models import ConsistencyEvaluator

                self.kg_evaluators[metric] = ConsistencyEvaluator(
                    graph_storage=self.graph_storage,
                    chunk_storage=self.chunk_storage,
                    llm_client=self.llm_client,
                )
            elif metric == "kg_structure":
                from graphgen.models import StructureEvaluator

                self.kg_evaluators[metric] = StructureEvaluator(
                    graph_storage=self.graph_storage,
                    **self.kwargs.get("structure_params", {}),
                )
            else:
                raise ValueError(f"Unknown QA metric: {metric}")

    async def _process_single_qa(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            qa_pair = QAPair(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
            )
            if not qa_pair.question or not qa_pair.answer:
                logger.error("Empty question or answer, skipping.")
                return {}
        except Exception as e:
            logger.error("Error in QAPair creation: %s", str(e))
            return {}

        for metric, evaluator in self.qa_evaluators.items():
            try:
                score = evaluator.evaluate(qa_pair)
                if isinstance(score, dict):
                    for sub_metric, sub_score in score.items():
                        item[f"{metric}_{sub_metric}"] = float(sub_score)
                else:
                    item[metric] = float(score)
            except Exception as e:
                logger.error("Error in %s evaluation: %s", metric, str(e))
                item[metric] = None
        return item

    def _evaluate_qa(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def transform_messages_format(items: list[dict]) -> list[dict]:
            """
            Transform from [{'messages': [...]}, ...] to [{'question': '...', 'answer': '...'}, ...]
            """
            transformed = []
            for item in items:
                messages = item.get("messages", [])
                question = next(
                    (m["content"] for m in messages if m.get("role") == "user"), ""
                )
                answer = next(
                    (m["content"] for m in messages if m.get("role") == "assistant"), ""
                )

                transformed.append({"question": question, "answer": answer})
            return transformed

        if not items:
            return []

        if not self.qa_evaluators:
            logger.warning("No QA evaluators initialized, skipping QA evaluation")
            return []

        items = transform_messages_format(items)
        results = run_concurrent(
            self._process_single_qa,
            items,
            desc="Evaluating QA items",
            unit="item",
        )

        results = [item for item in results if item]
        return results

    def _evaluate_kg(self) -> Dict[str, Any]:
        results = {}

        for metric, evaluator in self.kg_evaluators.items():
            try:
                logger.info("Running %s evaluation...", metric)
                score = evaluator.evaluate()
                results[metric] = score
            except Exception as e:
                logger.error("Error in %s evaluation: %s", metric, str(e))
                results[metric] = {"error": str(e)}
        return results

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        # QA evaluation
        if len(self.qa_evaluators) > 0:
            items = batch.to_dict(orient="records")
            results = self._evaluate_qa(items)
            return pd.DataFrame(results)

        # KG evaluation
        if len(self.kg_evaluators) > 0:
            results = self._evaluate_kg()
            # Convert dict to DataFrame (single row)
            return pd.DataFrame([results])

        # No metrics specified
        logger.warning("No metrics specified, returning empty DataFrame")
        return pd.DataFrame()
