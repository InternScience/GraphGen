from typing import List

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.bases.datatypes import Chunk
from graphgen.common import init_llm, init_storage
from graphgen.utils import logger

from .build_mm_kg import build_mm_kg
    def __init__(self, working_dir: str = "cache"):
        super().__init__(working_dir=working_dir, op_name="build_kg_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend="kuzu", working_dir=working_dir, namespace="graph"
        )

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        docs = batch.to_dict(orient="records")
        docs = [Chunk.from_dict(doc["_chunk_id"], doc) for doc in docs]

        # consume the chunks and build kg
        self.build_kg(docs)
        return pd.DataFrame([{"status": "kg_building_completed"}])

    def build_kg(self, chunks: List[Chunk]) -> None:
        """
        Build knowledge graph (KG) and merge into kg_instance
        """
        text_chunks = [chunk for chunk in chunks if chunk.type == "text"]
        mm_chunks = [
            chunk
            for chunk in chunks
            if chunk.type in ("image", "video", "table", "formula")
        ]
        if len(omics_chunks) == 0:
            logger.info("All omics chunks are already in the storage")
        else:
            logger.info(
                "[Omics Entity and Relation Extraction] processing %d chunks (DNA/RNA/protein)...",
                len(omics_chunks)
            )
            build_omics_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=omics_chunks,
            )

        self.graph_storage.index_done_callback()
