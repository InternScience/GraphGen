"""Pure stdlib contracts for SAG-adapted event extraction (testable without GraphGen)."""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Optional

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
_ALLOWED_TYPES = {item["type"] for item in PMS_ENTITY_TYPES}
_ALLOWED_EVENT_KEYS = {"title", "content", "entities", "is_valid"}
_ALLOWED_ENTITY_KEYS = {"type", "name", "description"}


def numbers_in(text: str) -> set[str]:
    """原文/事件中的数字集合（SAG grounding 的确定性口径）。"""
    return set(_NUMBER_RE.findall(text or ""))


def event_anchor(content: str) -> str | None:
    """事件的完整项目编号锚点；取首个出现的编号。"""
    match = PROJECT_ID_RE.search(content or "")
    return match.group(0) if match else None


def event_node_name(anchor: str, title: str, content: str) -> str:
    """确定性事件节点 id：event:{anchor}:{hash12}（内容 SHA，重叠 chunk 天然去重）。"""
    digest = hashlib.sha256(f"{anchor}\n{title}\n{content}".encode("utf-8")).hexdigest()[:12]
    return f"event:{anchor}:{digest}"


def parse_extraction_response(response: str, *, chunk_content: str) -> tuple[list[dict], dict[str, int]]:
    """解析并校验 LLM 事件抽取响应，返回 (有效事件列表, 丢弃计数)。

    严格合同（SAG schema.py 口径）：顶层/事件/实体禁合同外字段；is_valid=False、
    无项目锚点、数字不接地、实体类型越界的事件整条丢弃并计数。
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
    if not isinstance(payload, dict) or set(payload.keys()) != {"type", "data"}:
        dropped["response_contract_invalid"] += 1
        return [], dict(dropped)
    data = payload.get("data")
    if not isinstance(data, dict) or set(data.keys()) != {"items"}:
        dropped["response_contract_invalid"] += 1
        return [], dict(dropped)

    evidence_numbers = numbers_in(chunk_content)
    valid_events: list[dict] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            dropped["event_not_object"] += 1
            continue
        extra_keys = set(item.keys()) - _ALLOWED_EVENT_KEYS
        if extra_keys:
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
        anchor = event_anchor(content)
        if anchor is None:
            dropped["event_missing_anchor"] += 1
            continue
        ungrounded = {n for n in numbers_in(f"{title} {content}") if n not in evidence_numbers}
        if ungrounded:
            dropped["event_number_not_grounded"] += 1
            continue
        entities: list[dict] = []
        for entity in entities_raw:
            if not isinstance(entity, dict):
                dropped["entity_not_object"] += 1
                continue
            if set(entity.keys()) - _ALLOWED_ENTITY_KEYS:
                dropped["entity_extra_fields"] += 1
                continue
            etype = str(entity.get("type") or "").strip().upper()
            ename = str(entity.get("name") or "").strip()
            edesc = str(entity.get("description") or "").strip()
            if not etype or not ename or not edesc:
                dropped["entity_blank"] += 1
                continue
            if etype not in _ALLOWED_TYPES:
                dropped["entity_type_unknown"] += 1
                continue
            entities.append({"type": etype, "name": ename, "description": edesc})
        valid_events.append(
            {"title": title, "content": content, "anchor": anchor, "entities": entities}
        )
    return valid_events, dict(dropped)

