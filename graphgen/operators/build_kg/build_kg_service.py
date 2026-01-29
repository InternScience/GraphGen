from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.bases.datatypes import Chunk
from graphgen.common import init_llm, init_storage
from graphgen.utils import logger

from .build_mm_kg import build_mm_kg
from .build_text_kg import build_text_kg


class BuildKGService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **build_kwargs
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="build_kg"
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )
        self.build_kwargs = build_kwargs
        self.max_loop: int = int(self.build_kwargs.get("max_loop", 3))

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Build knowledge graph (KG) and merge into kg_instance
        """
        chunks = [Chunk.from_dict(doc["_trace_id"], doc) for doc in batch]
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

        meta_updates = {}
        results = []
        for node in nodes:
            if not node:
                continue
            trace_id = node["entity_name"]
            results.append(
                {
                    "_trace_id": trace_id,
                    "node": node,
                    "edge": {},
                }
            )
            source_ids = node.get("source_id", "").split("<SEP>")
            for source_id in source_ids:
                meta_updates.setdefault(source_id, []).append(trace_id)
        for edge in edges:
            if not edge:
                continue
            trace_id = frozenset((edge["src_id"], edge["tgt_id"]))
            results.append(
                {
                    "_trace_id": str(trace_id),
                    "node": {},
                    "edge": edge,
                }
            )
            source_ids = edge.get("source_id", "").split("<SEP>")
            for source_id in source_ids:
                meta_updates.setdefault(source_id, []).append(str(trace_id))
        return results, meta_updates
