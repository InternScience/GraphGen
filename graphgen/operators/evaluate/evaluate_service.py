from typing import Any
import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator, QAPair
from graphgen.common import init_llm
from graphgen.utils import run_concurrent


class EvaluateService(BaseOperator):
    """
    1. KG Quality Evaluation
    2. QA Quality Evaluation
    """

    def __init__(self, working_dir: str = "cache", metrics: list[str] = None, **kwargs):
        super().__init__(working_dir=working_dir, op_name="evaluate_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.metrics = metrics
        self.kwargs = kwargs
        self.evaluators = {}

    def _init_evaluators(self):
        for metric in self.metrics:
            if metric == "qa_length":
                from graphgen.models import LengthEvaluator
                self.evaluators[metric] = LengthEvaluator()
            elif metric == "qa_mtld":
                from graphgen.models import MTLDEvaluator
                self.evaluators[metric] = MTLDEvaluator(self.kwargs.get("mtld_params", {}))
            elif metric == "qa_reward_score":
                from graphgen.models import RewardEvaluator
                self.evaluators[metric] = RewardEvaluator(self.kwargs.get("reward_params", {}))
            elif metric == "qa_uni_score":
                from graphgen.models import UniEvaluator
                self.evaluators[metric] = UniEvaluator(self.kwargs.get("uni_params", {}))
            else:
                raise ValueError(f"Unknown metric: {metric}")

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        items = batch.to_dict(orient="records")
        return pd.DataFrame(self.evaluate(items))

    async def _process_single(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            qa_pair = QAPair(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", ""))
            )
            if not qa_pair.question or not qa_pair.answer:
                self.logger.error("Empty question or answer, skipping.")
                return {}
        except Exception as e:
            self.logger.error(
                "Error in QAPair creation: %s",
                str(e)
            )
            return {}

        for metric, evaluator in self.evaluators.items():
            try:
                score = evaluator.evaluate(qa_pair)
                if isinstance(score, dict):
                    for sub_metric, sub_score in score.items():
                        item[f"{metric}_{sub_metric}"] = float(sub_score)
                else:
                    item[metric] = float(score)
            except Exception as e:
                self.logger.error(
                    "Error in %s evaluation: %s",
                    metric,
                    str(e)
                )
                item[metric] = None

    def evaluate(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []

        results = run_concurrent(
            self._process_single,
            items,
            desc="Evaluating items",
            unit="item",
        )

        results = [item for item in results if item]

        return results
