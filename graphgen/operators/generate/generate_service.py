import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm
from graphgen.models import (
    AggregatedGenerator,
    AtomicGenerator,
    CoTGenerator,
    MultiHopGenerator,
    OmicsQAGenerator,
    VQAGenerator,
)
from graphgen.utils import logger, run_concurrent


class GenerateService(BaseOperator):
    """
    Generate question-answer pairs based on nodes and edges.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        method: str = "aggregated",
        data_format: str = "ChatML",
    ):
        super().__init__(working_dir=working_dir, op_name="generate_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")

        self.method = method
        self.data_format = data_format

        if self.method == "atomic":
            self.generator = AtomicGenerator(self.llm_client)
        elif self.method == "aggregated":
            self.generator = AggregatedGenerator(self.llm_client)
        elif self.method == "multi_hop":
            self.generator = MultiHopGenerator(self.llm_client)
        elif self.method == "cot":
            self.generator = CoTGenerator(self.llm_client)
        elif self.method == "omics_qa":
            self.generator = OmicsQAGenerator(self.llm_client)
        elif self.method in ["vqa"]:
            self.generator = VQAGenerator(self.llm_client)
        else:
            raise ValueError(f"Unsupported generation mode: {method}")

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        items = batch.to_dict(orient="records")
        return pd.DataFrame(self.generate(items))

    def generate(self, items: list[dict]) -> list[dict]:
        """
        Generate question-answer pairs based on nodes and edges.
        :param items
        :return: QA pairs
        """
        logger.info("[Generation] mode: %s, batches: %d", self.method, len(items))
        items = [(item["nodes"], item["edges"]) for item in items]
        results = run_concurrent(
            self.generator.generate,
            items,
            desc="[4/4]Generating QAs",
            unit="batch",
        )

        # Filter out empty results
        results = [res for res in results if res]

        results = self.generator.format_generation_results(
            results, output_data_format=self.data_format
        )

        return results
