from typing import Tuple

from graphgen.bases import BaseOperator

from .tree_utils import infer_title_level


class HierarchyGenerateService(BaseOperator):
    """Assign title levels for component-pack records without changing the legacy pipeline."""

    def __init__(self, working_dir: str = "cache", kv_backend: str = "rocksdb"):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="hierarchy_generate",
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        results = []
        meta_updates = {}

        for doc in batch:
            components = doc.get("components", [])
            leveled_components = []
            for component in components:
                title = component.get("title", "")
                component = dict(component)
                component["title_level"] = int(
                    component.get("title_level") or infer_title_level(title)
                )
                leveled_components.append(component)

            result = {
                "type": doc.get("type", "component_pack"),
                "source_trace_id": doc.get("source_trace_id", ""),
                "components": leveled_components,
                "metadata": doc.get("metadata", {}),
            }
            result["_trace_id"] = self.get_trace_id(result)
            results.append(result)
            meta_updates.setdefault(doc["_trace_id"], []).append(result["_trace_id"])

        return results, meta_updates
