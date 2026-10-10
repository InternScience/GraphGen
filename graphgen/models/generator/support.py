"""Parse support blocks and build a display-safe graph view for generators."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_SUPPORT_RE = re.compile(r"<support>(.*?)</support>", re.DOTALL)


def parse_support(response: str) -> Optional[dict[str, Any]]:
    """Parse support XML; malformed or absent blocks are ignored."""
    match = _SUPPORT_RE.search(response or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"cited"}:
        return None
    cited = payload.get("cited")
    if not isinstance(cited, list) or not all(isinstance(value, str) and value.strip() for value in cited):
        return None
    return {"cited": [value.strip() for value in cited]}


def _is_event_node(node: tuple[str, dict]) -> bool:
    return str(node[1].get("graph_role", "")).lower() == "event"


def _event_display_name(node: tuple[str, dict]) -> str:
    title = str(node[1].get("description", "")).split("：", 1)[0].strip()
    return title or "事件"


def build_generation_view(item: dict[str, Any]) -> dict[str, Any]:
    """Replace internal event IDs with readable names for generation only.

    Canonical node/edge records remain unchanged for lineage. Duplicate event
    titles receive deterministic suffixes; support names are mapped back and
    validated against the original partition node IDs.
    """
    original_nodes = [node for node in item.get("nodes", []) if isinstance(node, (list, tuple)) and len(node) == 2]
    event_nodes = [node for node in original_nodes if _is_event_node(node)]
    used_names = {
        str(node[0]) for node in original_nodes if not _is_event_node(node)
    }
    display_by_id: dict[str, str] = {}
    for node in sorted(event_nodes, key=lambda row: str(row[0])):
        base = _event_display_name(node)
        display = base
        suffix = 2
        while display in used_names:
            display = f"{base} ({suffix})"
            suffix += 1
        used_names.add(display)
        display_by_id[str(node[0])] = display

    nodes = [
        (display_by_id.get(str(node_id), node_id), data)
        for node_id, data in original_nodes
    ]
    edges = []
    support_aliases = {name: node_id for node_id, name in display_by_id.items()}
    for edge in item.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            continue
        source, target, data = edge
        display_source = display_by_id.get(str(source), source)
        display_target = display_by_id.get(str(target), target)
        edges.append((display_source, display_target, data))
        if str(source) in display_by_id:
            support_aliases[f"{display_source} - {display_target}"] = f"{source} - {target}"
        elif str(target) in display_by_id:
            support_aliases[f"{display_source} - {display_target}"] = f"{source} - {target}"

    def resolve_support(support: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not support:
            return None
        resolved = []
        for cited in support.get("cited", []):
            if cited in support_aliases:
                resolved.append(support_aliases[cited])
            else:
                resolved.append(cited)
        return {"cited": resolved}

    return {
        "nodes": nodes,
        "edges": edges,
        "canonical_nodes": [str(node_id) for node_id, _ in original_nodes],
        "resolve_support": resolve_support,
    }


def validate_support(support: Optional[dict[str, Any]], node_names: set[str]) -> Optional[dict[str, Any]]:
    """Keep support only when all citations point to existing partition nodes."""
    if not support:
        return None
    cited = support.get("cited") or []
    if cited and set(cited) <= node_names:
        return support
    return None
