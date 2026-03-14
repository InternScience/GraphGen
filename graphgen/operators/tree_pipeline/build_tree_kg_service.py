from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.bases.datatypes import Chunk
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger

from graphgen.operators.build_kg.build_mm_kg import build_mm_kg
from graphgen.operators.build_kg.build_text_kg import build_text_kg


class BuildTreeKGService(BaseOperator):
    """Build KG from tree-aware chunks while keeping output format compatible."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **build_kwargs,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="build_tree_kg",
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.build_kwargs = build_kwargs
        self.max_loop: int = int(self.build_kwargs.get("max_loop", 3))

    @staticmethod
    def _inject_tree_context(chunk: Chunk) -> Chunk:
        metadata = dict(chunk.metadata or {})
        path = metadata.get("metadata", {}).get("path") if isinstance(metadata.get("metadata"), dict) else metadata.get("path")
        if not path:
            return chunk

        contextual_content = f"[Document Path]\n{path}\n\n[Chunk]\n{chunk.content}"
        return Chunk(
            id=chunk.id,
            content=contextual_content,
            type=chunk.type,
            metadata=metadata,
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        chunks = [Chunk.from_dict(doc["_trace_id"], doc) for doc in batch]
        text_chunks = [self._inject_tree_context(chunk) for chunk in chunks if chunk.type == "text"]
        mm_chunks = [
            chunk
            for chunk in chunks
            if chunk.type in ("image", "video", "table", "formula")
        ]

        nodes = []
        edges = []

        if text_chunks:
            logger.info("[Tree Text Entity and Relation Extraction] processing ...")
            text_nodes, text_edges = build_text_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=text_chunks,
                max_loop=self.max_loop,
            )
            nodes += text_nodes
            edges += text_edges
        else:
            logger.info("All tree text chunks are already in the storage")

        if mm_chunks:
            logger.info("[Tree Multi-modal Entity and Relation Extraction] processing ...")
            mm_nodes, mm_edges = build_mm_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=mm_chunks,
            )
            nodes += mm_nodes
            edges += mm_edges
        else:
            logger.info("All tree multi-modal chunks are already in the storage")

        self.graph_storage.index_done_callback()
        meta_updates = {}
        results = []

        for node in nodes:
            if not node:
                continue
            trace_id = node["entity_name"]
            results.append({"_trace_id": trace_id, "node": node, "edge": {}})
            for source_id in node.get("source_id", "").split("<SEP>"):
                meta_updates.setdefault(source_id, []).append(trace_id)

        for edge in edges:
            if not edge:
                continue
            trace_id = str(frozenset((edge["src_id"], edge["tgt_id"])))
            results.append({"_trace_id": trace_id, "node": {}, "edge": edge})
            for source_id in edge.get("source_id", "").split("<SEP>"):
                meta_updates.setdefault(source_id, []).append(trace_id)

        return results, meta_updates
