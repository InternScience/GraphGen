"""<support> 审计块解析（PMS fork，Phase 2 升级 C）。

生成器可在 <question>/<answer> 之后输出
``<support>{"cited": ["节点或事件名称", ...]}</support>``；
本模块只负责解析，有效性校验（cited 是否真存在于出题分区）在
GenerateService.process 中按分区的节点集合执行，失败剥离 support 保留题面。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_SUPPORT_RE = re.compile(r"<support>(.*?)</support>", re.DOTALL)


def parse_support(response: str) -> Optional[dict[str, Any]]:
    """从模型响应解析 support 块；缺失/畸形返回 None（不影响题面解析）。"""
    match = _SUPPORT_RE.search(response or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload.keys()) != {"cited"}:
        return None
    cited = payload.get("cited")
    if not isinstance(cited, list) or not all(isinstance(c, str) and c.strip() for c in cited):
        return None
    return {"cited": [c.strip() for c in cited]}


def validate_support(support: Optional[dict[str, Any]], node_names: set[str]) -> Optional[dict[str, Any]]:
    """cited 全部存在于分区节点集合 → 原样返回；否则返回 None（剥离）。"""
    if not support:
        return None
    cited = support.get("cited") or []
    if cited and set(cited) <= node_names:
        return support
    return None
