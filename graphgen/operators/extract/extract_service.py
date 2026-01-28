import json

import pandas as pd

from graphgen.bases import BaseLLMWrapper, BaseOperator, Chunk
from graphgen.common import init_llm
from graphgen.models.extractor import SchemaGuidedExtractor
from graphgen.utils import logger, run_concurrent


class ExtractService(BaseOperator):
    def __init__(
        self, working_dir: str = "cache", kv_backend: str = "rocksdb", **extract_kwargs
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="extract_service"
        )
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

    def process(self, batch: list) -> pd.DataFrame:
        logger.info("Start extracting information from %d items", len(batch))
        chunks = [Chunk.from_dict(item["_trace_id"], item) for item in batch]
        results = run_concurrent(
            self.extractor.extract,
            chunks,
            desc="Extracting information",
            unit="item",
        )
        results = self.extractor.merge_extractions(results)

        results = [
            {"_extract_id": key, "extracted_data": value}
            for key, value in results.items()
        ]
        return results
