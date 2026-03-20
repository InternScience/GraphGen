import re
from typing import Dict, List, Tuple

from graphgen.bases import BaseOperator


def _normalize_section_key(title: str) -> str:
    base = (title or "Document").strip() or "Document"
    base = re.sub(r"^#{1,6}\s+", "", base)
    base = re.sub(r"\s+", "-", base)
    base = re.sub(r"[\\/]+", "-", base)
    base = re.sub(r"[^\w.\-]+", "-", base, flags=re.UNICODE)
    base = re.sub(r"-{2,}", "-", base).strip("-_.")
    return base or "document"


def _uniq_key(parent: Dict, base: str) -> str:
    base = (base or "document").strip() or "document"
    children = parent.setdefault("children", {})
    if base not in children:
        return base
    idx = 1
    while f"{base}_{idx}" in children:
        idx += 1
    return f"{base}_{idx}"


def _make_node_key(parent: Dict, node_type: str, title: str) -> str:
    if node_type == "section":
        return _uniq_key(parent, _normalize_section_key(title))
    return _uniq_key(parent, node_type or "text")


class TreeConstructService(BaseOperator):
    """Build a tree document structure from leveled components."""

    def __init__(self, working_dir: str = "cache", kv_backend: str = "rocksdb"):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="tree_construct",
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        results = []
        meta_updates = {}

        for doc in batch:
            root = {
                "node_id": "root",
                "title": "root",
                "level": 0,
                "content": "",
                "node_type": "root",
                "path": "root",
                "children": {},
            }
            section_stack: List[Tuple[Dict, int]] = [(root, 0)]
            ordered_nodes: List[Dict] = []

            for idx, component in enumerate(doc.get("components", []), start=1):
                level = max(1, int(component.get("title_level", 1)))
                node_type = component.get("type", "text")
                node = {
                    "node_id": f"n{idx}",
                    "title": component.get("title", "Document"),
                    "level": level,
                    "content": component.get("content", ""),
                    "node_type": node_type,
                    "metadata": dict(component.get("metadata", {})),
                    "children": {},
                }

                if node_type == "section":
                    while section_stack and section_stack[-1][1] >= level:
                        section_stack.pop()
                    parent = section_stack[-1][0] if section_stack else root
                else:
                    parent = section_stack[-1][0] if section_stack else root

                key = _make_node_key(parent, node_type, node["title"])
                parent["children"][key] = node
                parent_path = parent.get("path", "root")
                node["path"] = f"{parent_path}/{key}"
                node["parent_id"] = parent.get("node_id", "root")
                ordered_nodes.append(node)
                if node_type == "section":
                    section_stack.append((node, level))

            result = {
                "type": "doc_tree",
                "source_trace_id": doc.get("source_trace_id", ""),
                "tree": root,
                "tree_nodes": ordered_nodes,
                "metadata": doc.get("metadata", {}),
            }
            result["_trace_id"] = self.get_trace_id(result)
            results.append(result)
            meta_updates.setdefault(doc["_trace_id"], []).append(result["_trace_id"])

        return results, meta_updates
