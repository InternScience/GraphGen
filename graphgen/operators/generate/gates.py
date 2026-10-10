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


def apply_output_gate(name: str, qa_pairs: Any) -> list[dict]:
    """Apply a registered gate by name; fail-closed on unknown gates."""
    if name not in GATES:
        raise ValueError(f"output_gate_not_registered:{name}")
    return GATES[name](qa_pairs)
