"""Pure stdlib contracts for profile-driven SAG event extraction."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

PMS_ENTITY_TYPES: list[dict] = [
    {"type": "PROJECT", "description": "项目本体，含完整项目编号"},
    {"type": "PERSON", "description": "人员，项目成员或干系人"},
    {"type": "ROLE", "description": "项目内职务或角色"},
    {"type": "ORGANIZATION", "description": "组织、部门、公司或团队"},
    {"type": "PHASE", "description": "项目阶段或状态"},
    {"type": "LOCATION", "description": "地点或区域"},
    {"type": "DATE", "description": "日期或时间"},
    {"type": "METRIC", "description": "数量、比例等指标"},
]
PROJECT_ID_RE = re.compile(r"\b[PI]-\d{6,}\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![A-Za-z0-9_.])")
_ALLOWED_EVENT_KEYS = {"title", "content", "entities", "is_valid"}
_ALLOWED_ENTITY_KEYS = {"type", "name", "description"}


def numbers_in(text: str) -> set[str]:
    """Return numeric strings using the deterministic SAG grounding policy."""
    return set(_NUMBER_RE.findall(text or ""))


def event_anchor(content: str, pattern: str | None = None) -> str | None:
    """Return the first configured anchor match, if any."""
    if pattern is None:
        return None
    match = re.compile(pattern).search(content or "")
    if match is None:
        return None
    if match.lastindex:
        return match.group(1)
    return match.group(0)


def event_node_name(anchor: str | None, title: str, content: str, *, legacy: bool = True) -> str:
    """Create a deterministic node ID; legacy PMS mode preserves persisted IDs."""
    if legacy:
        digest = hashlib.sha256(f"{anchor}\n{title}\n{content}".encode("utf-8")).hexdigest()[:12]
        return f"event:{anchor}:{digest}"
    digest = hashlib.sha256(
        json.dumps([anchor, title, content], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return f"event:{digest}"


def parse_extraction_response(
    response: str,
    *,
    chunk_content: str,
    entity_types: list[dict] | None = None,
    anchor_pattern: str | None = None,
    anchor_required: bool | None = None,
    number_grounding: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """Parse the strict event response contract under an explicit profile policy.

    Omitted policy arguments retain the historical PMS behavior for callers that
    import this pure-stdlib helper directly.
    """
    dropped: dict[str, int] = defaultdict(int)
    text = (response or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        dropped["response_not_json"] += 1
        return [], dict(dropped)
    if not isinstance(payload, dict) or set(payload) != {"type", "data"}:
        dropped["response_contract_invalid"] += 1
        return [], dict(dropped)
    data = payload.get("data")
    if not isinstance(data, dict) or set(data) != {"items"} or not isinstance(data["items"], list):
        dropped["response_contract_invalid"] += 1
        return [], dict(dropped)

    profile_policy = entity_types is not None or anchor_pattern is not None or anchor_required is not None
    if profile_policy:
        types = entity_types or []
        allowed_types = {str(item["type"]).strip().upper() for item in types}
        if anchor_required is None:
            anchor_required = True
    else:
        types = PMS_ENTITY_TYPES
        allowed_types = {item["type"] for item in types}
        anchor_pattern = PROJECT_ID_RE.pattern
        anchor_required = True
    evidence_numbers = numbers_in(chunk_content) if number_grounding else set()
    valid_events: list[dict] = []
    for item in data["items"]:
        if not isinstance(item, dict):
            dropped["event_not_object"] += 1
            continue
        if set(item) - _ALLOWED_EVENT_KEYS:
            dropped["event_extra_fields"] += 1
            continue
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        entities_raw = item.get("entities") or []
        if not title or not content:
            dropped["event_blank"] += 1
            continue
        if item.get("is_valid") is False:
            dropped["event_self_invalid"] += 1
            continue
        anchor = event_anchor(content, anchor_pattern)
        if anchor_required and anchor is None:
            dropped["event_missing_anchor"] += 1
            continue
        if number_grounding and {n for n in numbers_in(f"{title} {content}") if n not in evidence_numbers}:
            dropped["event_number_not_grounded"] += 1
            continue
        entities: list[dict] = []
        if not isinstance(entities_raw, list):
            dropped["event_entities_invalid"] += 1
            continue
        for entity in entities_raw:
            if not isinstance(entity, dict):
                dropped["entity_not_object"] += 1
                continue
            if set(entity) - _ALLOWED_ENTITY_KEYS:
                dropped["entity_extra_fields"] += 1
                continue
            etype = str(entity.get("type") or "").strip().upper()
            ename = str(entity.get("name") or "").strip()
            edesc = str(entity.get("description") or "").strip()
            if not etype or not ename or not edesc:
                dropped["entity_blank"] += 1
                continue
            if etype not in allowed_types:
                dropped["entity_type_unknown"] += 1
                continue
            entities.append({"type": etype, "name": ename, "description": edesc})
        valid_events.append({"title": title, "content": content, "anchor": anchor, "entities": entities})
    return valid_events, dict(dropped)
