from typing import Any, Dict, List, Tuple

from graphgen.bases import BaseGraphStorage


class ConsistencyEvaluator:
    """Evaluates consistency by detecting attribute value conflicts."""

    def __init__(self, graph_storage: BaseGraphStorage):
        self.graph_storage = graph_storage

    def evaluate(self) -> Dict[str, Any]:
        all_nodes = self.graph_storage.get_all_nodes() or []
        if not all_nodes:
            return {"error": "Empty graph"}

        conflicts = []
        conflict_entities = set()

        for node_id, node_data in all_nodes:
            if not isinstance(node_data, dict):
                continue

            # Check each attribute for multiple values
            for attr_key, attr_value in node_data.items():
                # Skip special keys
                if attr_key.startswith("_") or attr_key in ["id", "loss"]:
                    continue

                # If attribute has multiple values (list), check for conflicts
                if isinstance(attr_value, list):
                    unique_values = set(str(v) for v in attr_value if v)
                    if len(unique_values) > 1:
                        conflicts.append(
                            {
                                "entity_id": node_id,
                                "attribute": attr_key,
                                "values": list(unique_values),
                            }
                        )
                        conflict_entities.add(node_id)

        total_entities = len(all_nodes)
        conflict_rate = (
            len(conflict_entities) / total_entities if total_entities > 0 else 0
        )

        return {
            "conflict_rate": conflict_rate,
            "conflict_entities_count": len(conflict_entities),
            "total_entities": total_entities,
            "conflicts": conflicts[:100],  # Limit to first 100 conflicts
        }
