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
    ):
        super().__init__(working_dir=working_dir, kv_backend=kv_backend, op_name="tree_chunk")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def process(self, batch: list) -> Tuple[list, dict]:
        results = []
        meta_updates = {}

        for doc in batch:
            input_trace_id = doc["_trace_id"]
            source_trace_id: Optional[str] = doc.get("source_trace_id")
            for node in doc.get("tree_nodes", []):
                content = node.get("content", "")
                if not content:
                    continue
                language = detect_main_language(content)
                chunks = split_chunks(
                    content,
                    language=language,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                for chunk_text in chunks:
                    row = {
                        "content": chunk_text,
                        "type": node.get("node_type", "text"),
                        "metadata": {
                            "language": language,
                            "length": len(chunk_text),
                            "path": node.get("path", "root"),
                            "level": node.get("level", 1),
                            "node_id": node.get("node_id"),
                            "parent_id": node.get("parent_id"),
                            "source_trace_id": source_trace_id,
                        },
                    }
                    row["_trace_id"] = self.get_trace_id(row)
                    results.append(row)
                    meta_updates.setdefault(input_trace_id, []).append(row["_trace_id"])

        return results, meta_updates
