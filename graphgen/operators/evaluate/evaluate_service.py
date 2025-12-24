import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm


class EvaluateService(BaseOperator):
    """
    1. KG Quality Evaluation
    2. QA Quality Evaluation
    """

    def __init__(self, working_dir: str = "cache", metrics: list[str] = None):
        # optional 传入 graph
        super().__init__(working_dir=working_dir, op_name="evaluate_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.metrics = metrics or []

        self.evaluators = {
            "xxx": "xxxEvaluator"
        }

        self.graph_storage = init_storage(
            xx, xx, xx
        )

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        items = batch.to_dict(orient="records")
        return pd.DataFrame(self.evaluate(items))

    def evaluate(self, items: list[dict]) -> list[dict]:
        # 用evaluators 评估 items
        pass
