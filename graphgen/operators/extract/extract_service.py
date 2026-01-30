import json
from typing import Tuple

from graphgen.bases import BaseLLMWrapper, BaseOperator, Chunk
from graphgen.common.init_llm import init_llm
from graphgen.models.extractor import SchemaGuidedExtractor
from graphgen.utils import logger, run_concurrent


class ExtractService(BaseOperator):
    def __init__(
        self, working_dir: str = "cache", kv_backend: str = "rocksdb", **extract_kwargs
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="extract"
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

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Extract information from the batch of chunks.
        :return: A tuple of (results, meta_updates)
            results: A list of dicts containing extracted information. Each dict has the structure:
                {"_trace_id": str, "content": dict}
            meta_updates: A dict mapping source IDs to lists of trace IDs for the extracted information.
        """
        logger.info("Start extracting information from %d items", len(batch))
        chunks = [Chunk.from_dict(item["_trace_id"], item) for item in batch]
        results = run_concurrent(
            self.extractor.extract,
            chunks,
            desc="Extracting information",
            unit="item",
        )

        meta_updates = {}
        final_results = []
        # chunk -> extracted info
        for input_trace_id, result in zip(
            [item["_trace_id"] for item in batch], results
        ):
            if not result:
                continue
            result = {"_trace_id": self.get_trace_id(result), "content": result}
            meta_updates.setdefault(input_trace_id, []).append(result["_trace_id"])
            final_results.append(result)
        return final_results, meta_updates
