from typing import Optional, Tuple

from graphgen.bases import BaseOperator
from graphgen.operators.chunk.chunk_service import split_chunks
from graphgen.utils import detect_main_language


class TreeChunkService(BaseOperator):
    """Chunk tree nodes into path-aware chunks for downstream KG building."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        split_text_nodes: bool = True,
    ):
        super().__init__(working_dir=working_dir, kv_backend=kv_backend, op_name="tree_chunk")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        if isinstance(split_text_nodes, str):
            split_text_nodes = split_text_nodes.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        self.split_text_nodes = bool(split_text_nodes)

    def process(self, batch: list) -> Tuple[list, dict]:
        results = []
        meta_updates = {}

        for doc in batch:
            input_trace_id = doc["_trace_id"]
            source_trace_id: Optional[str] = doc.get("source_trace_id")
            for node in doc.get("tree_nodes", []):
                content = node.get("content", "")
                node_type = node.get("node_type", "text")
                if node_type == "section":
                    continue
                if not content and node_type == "text":
                    continue
                node_metadata = dict(node.get("metadata", {}))
                language = detect_main_language(content) if content else "en"
                if node_type == "text" and self.split_text_nodes:
                    chunks = split_chunks(
                        content,
                        language=language,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                else:
                    chunks = [content]
                for chunk_text in chunks:
                    metadata = {
                        "language": language,
                        "length": len(chunk_text),
                        "path": node.get("path", "root"),
                        "level": node.get("level", 1),
                        "node_id": node.get("node_id"),
                        "parent_id": node.get("parent_id"),
                        "source_trace_id": source_trace_id,
                    }
                    metadata.update(node_metadata)
                    row = {
                        "content": chunk_text,
                        "type": node_type,
                        "metadata": metadata,
                    }
                    row["_trace_id"] = self.get_trace_id(row)
                    results.append(row)
                    meta_updates.setdefault(input_trace_id, []).append(row["_trace_id"])

        return results, meta_updates
