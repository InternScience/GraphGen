import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm
from graphgen.models import (
    AggregatedGenerator,
    AtomicGenerator,
    CoTGenerator,
    MultiHopGenerator,
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

        results = self.generator.format_generation_results(
            results, output_data_format=self.data_format
        )

        return results
