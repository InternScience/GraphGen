from typing import List

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.bases.datatypes import Chunk
from graphgen.common import init_llm, init_storage
from graphgen.utils import logger

from .build_mm_kg import build_mm_kg
from .build_text_kg import build_text_kg


class BuildKGService(BaseOperator):
    def __init__(
        self, working_dir: str = "cache", graph_backend: str = "kuzu", **build_kwargs
    ):
        super().__init__(working_dir=working_dir, op_name="build_kg_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )
        self.build_kwargs = build_kwargs
        self.max_loop: int = int(self.build_kwargs.get("max_loop", 3))

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        docs = batch.to_dict(orient="records")
        docs = [Chunk.from_dict(doc["_chunk_id"], doc) for doc in docs]

        # consume the chunks and build kg
        nodes, edges = self.build_kg(docs)
        return pd.DataFrame(
            [{"node": node, "edge": []} for node in nodes]
            + [{"node": [], "edge": edge} for edge in edges]
        )

    def build_kg(self, chunks: List[Chunk]) -> tuple:
        """
        Build knowledge graph (KG) and merge into kg_instance
        """
        text_chunks = [chunk for chunk in chunks if chunk.type == "text"]
        mm_chunks = [
            chunk
            for chunk in chunks
            if chunk.type in ("image", "video", "table", "formula")
        ]

        nodes = []
        edges = []

        if len(text_chunks) == 0:
            logger.info("All text chunks are already in the storage")
        else:
            logger.info("[Text Entity and Relation Extraction] processing ...")
            text_nodes, text_edges = build_text_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=text_chunks,
                max_loop=self.max_loop,
            )
            nodes += text_nodes
            edges += text_edges
        if len(mm_chunks) == 0:
            logger.info("All multi-modal chunks are already in the storage")
        else:
            logger.info("[Multi-modal Entity and Relation Extraction] processing ...")
            mm_nodes, mm_edges = build_mm_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=mm_chunks,
            )
            nodes += mm_nodes
            edges += mm_edges

        self.graph_storage.index_done_callback()
        logger.info("Knowledge graph building completed.")

        return nodes, edges
