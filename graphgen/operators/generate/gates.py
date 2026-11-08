"""Deterministic output gates for generated QA pairs.

A gate is a pure, synchronous filter applied to the parsed QA list of one
generate call before results reach the native cache. Gates never modify or
fabricate model output; they only drop pairs that violate a declared
deterministic contract. Gates are registered by name via
:func:`register_gate` and selected by name through ``GenerateService``.
"""

from __future__ import annotations

from typing import Any, Callable

GATES: dict[str, Callable[[Any], list[dict]]] = {}


def register_gate(name: str) -> Callable:
    """Register a gate function under a unique name."""

    def _decorator(func: Callable[[Any], list[dict]]) -> Callable[[Any], list[dict]]:
        if name in GATES:
            raise ValueError(f"output_gate_already_registered:{name}")
        GATES[name] = func
        return func

    return _decorator


@register_gate("non_empty_answer")
def non_empty_answer(qa_pairs: Any) -> list[dict]:
    """Generic built-in gate: keep only pairs with a non-empty answer."""
    if not isinstance(qa_pairs, list):
        return []
    return [
        pair
        for pair in qa_pairs
        if isinstance(pair, dict)
        and isinstance(pair.get("question"), str)
        and isinstance(pair.get("answer"), str)
        and pair["answer"].strip()
    ]


def apply_output_gate(name: str, qa_pairs: Any) -> list[dict]:
    """Apply a registered gate by name; fail-closed on unknown gates."""
    if name not in GATES:
        raise ValueError(f"output_gate_not_registered:{name}")
    return GATES[name](qa_pairs)
