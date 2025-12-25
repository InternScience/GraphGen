import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm


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

    def evaluate(self, items: list[dict]) -> list[dict]:
        print(items)
        pass

