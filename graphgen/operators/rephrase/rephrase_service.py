import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm
from graphgen.utils import run_concurrent


class RephraseService(BaseOperator):
    """
    Generate question-answer pairs based on nodes and edges.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        method: str = "aggregated",
        **rephrase_kwargs,
    ):
        super().__init__(working_dir=working_dir, op_name="rephrase_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.method = method
        self.rephrase_kwargs = rephrase_kwargs

        if self.method == "style_controlled":
            from graphgen.models import StyleControlledRephraser

            self.rephraser = StyleControlledRephraser(
                self.llm_client,
                style=rephrase_kwargs.get("style", "critical_analysis"),
            )
        else:
            raise ValueError(f"Unsupported rephrase method: {self.method}")

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        items = batch.to_dict(orient="records")
        return pd.DataFrame(self.rephrase(items))

    def rephrase(self, items: list[dict]) -> list[dict]:
        results = run_concurrent(
            self.rephraser.rephrase,
            items,
            desc="Rephrasing texts",
            unit="batch",
        )

        # Filter out empty results
        results = [res for res in results if res]
        return results
