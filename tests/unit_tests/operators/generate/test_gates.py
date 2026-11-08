"""Tests for the generic output-gate registry."""

import pytest

from graphgen.operators.generate.gates import GATES, apply_output_gate, register_gate


def test_builtin_gate_keeps_only_non_empty_answers():
    gate = GATES["non_empty_answer"]
    pairs = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "   "},
        {"question": "q3"},
        "not-a-dict",
    ]
    assert gate(pairs) == [{"question": "q1", "answer": "a1"}]
    assert gate(None) == []


def test_apply_unknown_gate_fails_closed():
    with pytest.raises(ValueError, match="output_gate_not_registered"):
        apply_output_gate("no_such_gate", [])


def test_register_duplicate_gate_fails_closed():
    @register_gate("dup_gate_test")
    def _gate(qa_pairs):
        return []

    with pytest.raises(ValueError, match="output_gate_already_registered"):

        @register_gate("dup_gate_test")
        def _other(qa_pairs):
            return []
