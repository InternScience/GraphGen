"""Deterministic output gates for generated QA pairs (PMS fork capability).

A gate is a pure, synchronous filter applied to the parsed QA list of one
generate call before results reach the native cache. Gates never modify or
fabricate model output; they only drop pairs that violate a declared
deterministic contract. See docs/PMS_PATCHES.md #2.
"""

from __future__ import annotations

import re
from typing import Any, Callable

GATES: dict[str, Callable[[Any], list[dict]]] = {}

_PROJECT_ID_PATTERN = re.compile(r"(?:P|I)-\d{6,}")

# 合同号核心形态（2026-10-28 按全库 102,347 个 contract_id 定版，见
# docs/2026-10-27-contract-info-query-data-source-best-practice.md D4）：
# 99.96% 命中 `YY-XX-NNNN`；汉字/`/Pxxxx`/W 等后缀由 PMS 侧"已知 ID 集合
# 最长精确匹配"解决，gate 只做存在性判定，不做精确抽取。
_CONTRACT_ID_PATTERN = re.compile(r"(?<![0-9A-Za-z-])\d{2}-[0-9A-Za-z]{1,6}-\d{3,}")

# v2.1 指代出题协议：片段不含合同号节点时允许“该合同”类指代提问，由 PMS
# mapper 在唯一 native 血缘下改写为真实编号（多血缘/零血缘在 PMS 侧拒绝）。
# gate 只做存在性判定，与合同号同层级。
_VAGUE_CONTRACT_PATTERN = re.compile(r"该合同|此合同|这份合同")


def register_gate(name: str) -> Callable:
    def _decorator(func: Callable[[Any], list[dict]]) -> Callable[[Any], list[dict]]:
        if name in GATES:
            raise ValueError(f"output_gate_already_registered:{name}")
        GATES[name] = func
        return func
    return _decorator


@register_gate("pms_project_anchor")
def pms_project_anchor(qa_pairs: Any) -> list[dict]:
    """PMS project-query anchor gate.

    Only pairs whose question carries a complete project id (P-/I- plus at
    least 6 digits) and a non-empty answer are executable by the project-query
    skill; name-only or refusal placeholder questions are dropped.
    """
    if not isinstance(qa_pairs, list):
        return []
    return [
        pair
        for pair in qa_pairs
        if isinstance(pair, dict)
        and isinstance(pair.get("question"), str)
        and bool(_PROJECT_ID_PATTERN.search(pair["question"]))
        and isinstance(pair.get("answer"), str)
        and pair["answer"].strip()
    ]


@register_gate("pms_contract_anchor")
def pms_contract_anchor(qa_pairs: Any) -> list[dict]:
    """PMS contract-info-query anchor gate.

    Passes pairs whose question carries a contract-id-shaped token (YY-XX-
    NNNN core, suffix-agnostic) or a vague contract demonstrative (“该合同”
    family, the v2.1 reference protocol that the PMS mapper rewrites with the
    unique lineage contract id), plus a non-empty answer. Name-only and
    refusal placeholder questions are dropped. Exact-id resolution against
    the snapshot happens PMS-side.
    """
    if not isinstance(qa_pairs, list):
        return []
    return [
        pair
        for pair in qa_pairs
        if isinstance(pair, dict)
        and isinstance(pair.get("question"), str)
        and (
            bool(_CONTRACT_ID_PATTERN.search(pair["question"]))
            or bool(_VAGUE_CONTRACT_PATTERN.search(pair["question"]))
        )
        and isinstance(pair.get("answer"), str)
        and pair["answer"].strip()
    ]


@register_gate("pms_contract_challenge")
def pms_contract_challenge(qa_pairs: Any) -> list[dict]:
    """PMS contract challenge 集合题门（batch1003 腿B）。

    在 ``pms_contract_anchor`` 基础上新增项目锚点出口：挑战 profile 的集合/
    聚合题以项目编号（P-/I- + ≥6 位）为锚点，问题不含合同号也不含“该合同”
    指代，现役门会把它们整条静默丢弃（表现为“模型出不了集合题”的假象）。
    本门放行三类锚点任一 + 非空答案；锚点真实性（编号存在于快照）仍由
    PMS 侧 mapper/审计把关，gate 只做形态存在性判定。
    """
    if not isinstance(qa_pairs, list):
        return []
    return [
        pair
        for pair in qa_pairs
        if isinstance(pair, dict)
        and isinstance(pair.get("question"), str)
        and (
            bool(_CONTRACT_ID_PATTERN.search(pair["question"]))
            or bool(_VAGUE_CONTRACT_PATTERN.search(pair["question"]))
            or bool(_PROJECT_ID_PATTERN.search(pair["question"]))
        )
        and isinstance(pair.get("answer"), str)
        and pair["answer"].strip()
    ]


def apply_output_gate(name: str, qa_pairs: Any) -> list[dict]:
    """Apply a registered gate by name; fail-closed on unknown gates."""
    if name not in GATES:
        raise ValueError(f"output_gate_not_registered:{name}")
    return GATES[name](qa_pairs)
