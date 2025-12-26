from typing import Any, Dict, List, Union

import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator, QAPair
from graphgen.common import init_llm
from graphgen.models import KGQualityEvaluator
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
        **kwargs
    ):
        super().__init__(working_dir=working_dir, op_name="evaluate_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.metrics = metrics or []
        self.kwargs = kwargs
        self.graph_backend = graph_backend
        self.kv_backend = kv_backend
        
        # Separate QA and KG metrics
        self.qa_metrics = [m for m in self.metrics if m.startswith("qa_")]
        self.kg_metrics = [m for m in self.metrics if m.startswith("kg_")]
        
        # Initialize evaluators
        self.qa_evaluators = {}
        self.kg_evaluator = None
        
        self._init_evaluators()

    def _init_evaluators(self):
        """Initialize QA and KG evaluators based on metrics."""
        # Initialize QA evaluators
        for metric in self.qa_metrics:
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
            else:
                raise ValueError(f"Unknown QA metric: {metric}")
        
        # Initialize KG evaluator if KG metrics are specified
        if self.kg_metrics:
            kg_params = self.kwargs.get("kg_params", {})
            self.kg_evaluator = KGQualityEvaluator(
                working_dir=self.working_dir,
                graph_backend=self.graph_backend,
                kv_backend=self.kv_backend,
                **kg_params
            )
            logger.info("KG evaluator initialized")

    async def _process_single(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            qa_pair = QAPair(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
            )
            if not qa_pair.question or not qa_pair.answer:
                self.logger.error("Empty question or answer, skipping.")
                return {}
        except Exception as e:
            self.logger.error("Error in QAPair creation: %s", str(e))
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
                self.logger.error("Error in %s evaluation: %s", metric, str(e))
                item[metric] = None
        return item

    @staticmethod
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

    def _evaluate_qa(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []

        if not self.qa_evaluators:
            logger.warning("No QA evaluators initialized, skipping QA evaluation")
            return []

        items = self.transform_messages_format(items)
        results = run_concurrent(
            self._process_single,
            items,
            desc="Evaluating QA items",
            unit="item",
        )

        results = [item for item in results if item]
        return results

    def _evaluate_kg(self) -> Dict[str, Any]:
        if not self.kg_evaluator:
            logger.warning("No KG evaluator initialized, skipping KG evaluation")
            return {}

        results = {}
        
        # Map metric names to evaluation functions
        kg_metric_map = {
            "kg_accuracy": self.kg_evaluator.evaluate_accuracy,
            "kg_consistency": self.kg_evaluator.evaluate_consistency,
            "kg_structure": self.kg_evaluator.evaluate_structure,
        }
        
        # Run KG evaluations based on metrics
        for metric in self.kg_metrics:
            if metric in kg_metric_map:
                logger.info("Running %s evaluation...", metric)
                metric_key = metric.replace("kg_", "")  # Remove "kg_" prefix
                try:
                    results[metric_key] = kg_metric_map[metric]()
                except Exception as e:
                    logger.error("Error in %s evaluation: %s", metric, str(e))
                    results[metric_key] = {"error": str(e)}
            else:
                logger.warning("Unknown KG metric: %s, skipping", metric)
        
        # If no valid metrics were found, run all evaluations
        if not results:
            logger.info("No valid KG metrics found, running all evaluations")
            results = self.kg_evaluator.evaluate_all()
        
        return results

    def evaluate(
        self, items: list[dict[str, Any]] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        # Determine evaluation type
        has_qa_metrics = len(self.qa_metrics) > 0
        has_kg_metrics = len(self.kg_metrics) > 0
        
        # If items provided and QA metrics exist, do QA evaluation
        if items is not None and has_qa_metrics:
            return self._evaluate_qa(items)
        
        # If KG metrics exist, do KG evaluation
        if has_kg_metrics:
            return self._evaluate_kg()
        
        # If no metrics specified, try to infer from context
        if items is not None:
            logger.warning("No QA metrics specified but items provided, skipping evaluation")
            return []
        else:
            logger.warning("No metrics specified, skipping evaluation")
            return {}

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        has_qa_metrics = len(self.qa_metrics) > 0
        has_kg_metrics = len(self.kg_metrics) > 0
        
        # QA evaluation: process batch items
        if has_qa_metrics:
            items = batch.to_dict(orient="records")
            results = self._evaluate_qa(items)
            return pd.DataFrame(results)
        
        # KG evaluation: evaluate from storage
        if has_kg_metrics:
            results = self._evaluate_kg()
            # Convert dict to DataFrame (single row)
            return pd.DataFrame([results])
        
        # No metrics specified
        logger.warning("No metrics specified, returning empty DataFrame")
        return pd.DataFrame()
