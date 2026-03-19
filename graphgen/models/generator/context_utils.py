from typing import Any


def _compact_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def format_node_context(index: int, node: tuple[str, dict]) -> str:
    node_id, node_data = node
    description = _compact_field(node_data.get("description", ""))
    evidence = _compact_field(node_data.get("evidence_span", ""))

    parts = [f"{index}. {node_id}: {description}"]
    if evidence:
        parts.append(f"   Evidence: {evidence}")
    return "\n".join(parts)


def format_edge_context(index: int, edge: tuple[Any, Any, dict]) -> str:
    src_id, tgt_id, edge_data = edge
    description = _compact_field(edge_data.get("description", ""))
    relation_type = _compact_field(edge_data.get("relation_type", ""))
    evidence = _compact_field(edge_data.get("evidence_span", ""))

    relation_label = f" [{relation_type}]" if relation_type else ""
    parts = [f"{index}. {src_id} -- {tgt_id}{relation_label}: {description}"]
    if evidence:
        parts.append(f"   Evidence: {evidence}")
    return "\n".join(parts)


def build_grounded_context(
    batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
) -> tuple[str, str]:
    nodes, edges = batch
    entities_str = "\n".join(
        format_node_context(index + 1, node) for index, node in enumerate(nodes)
    )
    relationships_str = "\n".join(
        format_edge_context(index + 1, edge) for index, edge in enumerate(edges)
    )
    return entities_str, relationships_str
