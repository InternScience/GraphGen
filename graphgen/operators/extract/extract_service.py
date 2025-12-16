import json

import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm
from graphgen.models.extractor import SchemaGuidedExtractor
from graphgen.utils import logger, run_concurrent


class ExtractService(BaseOperator):
    def __init__(self, working_dir: str = "cache", **extract_kwargs):
        super().__init__(working_dir=working_dir, op_name="extract_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.extract_kwargs = extract_kwargs
        self.method = self.extract_kwargs.get("method")
        if self.method == "schema_guided":
            schema_file = self.extract_kwargs.get("schema_path")
            with open(schema_file, "r", encoding="utf-8") as f:
                schema = json.load(f)
            self.extractor = SchemaGuidedExtractor(self.llm_client, schema)
        else:
            raise ValueError(f"Unsupported extraction method: {self.method}")

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        items = batch.to_dict(orient="records")
        return pd.DataFrame(self.extract(items))

    def extract(self, items: list[dict]) -> list[dict]:

        logger.info("Start extracting information from %d items", len(items))

        results = run_concurrent(
            self.extractor.extract,
            items,
            desc="Extracting information",
            unit="item",
        )
        results = self.extractor.merge_extractions(results)

        results = [
            {"_extract_id": key, "extracted_data": value}
            for key, value in results.items()
        ]
        return results
