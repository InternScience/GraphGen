"""Empty-response prefill retry must parse both backend continuation styles.

Regression for the #194 review nit: with ``assistant_prefill="{"`` some
backends genuinely continue the assistant turn and return only the text
*after* ``{``; feeding that to ``json.loads`` as-is still fails.
"""

import asyncio

from graphgen.bases import Chunk
from graphgen.models.extractor import SchemaGuidedExtractor

SCHEMA = {
    "name": "event document",
    "properties": {
        "event": {"type": "string", "description": "Name of the event."},
    },
    "required": ["event"],
}


class StubLLM:
    """Returns queued responses and records whether a prefill was requested."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_answer(self, prompt, assistant_prefill=None):
        self.calls.append(assistant_prefill)
        return self.responses.pop(0)


def _extract(llm):
    extractor = SchemaGuidedExtractor(llm, SCHEMA)
    return asyncio.run(extractor.extract(Chunk(id="c1", content="text", type="text")))


def test_prefill_retry_parses_continuation_without_brace(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", "1")
    llm = StubLLM(["", '"event": "launch" }'])
    assert _extract(llm) == {"event": "launch"}
    assert llm.calls == [None, "{"]


def test_prefill_retry_accepts_fresh_object_response(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", "1")
    llm = StubLLM(["", '{"event": "launch"}'])
    assert _extract(llm) == {"event": "launch"}
    assert llm.calls == [None, "{"]


def test_prefill_retry_tolerates_leading_whitespace(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", "1")
    llm = StubLLM(["", '  {"event": "launch"}'])
    assert _extract(llm) == {"event": "launch"}


def test_no_prefill_retry_when_first_response_parses(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", "1")
    llm = StubLLM(["{\"event\": \"launch\"}"])
    assert _extract(llm) == {"event": "launch"}
    assert llm.calls == [None]


def test_empty_response_stays_unparsed_when_gate_disabled(monkeypatch):
    monkeypatch.delenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", raising=False)
    llm = StubLLM([""])
    assert _extract(llm) == {}
    assert llm.calls == [None]


def test_failed_retry_still_fails_closed(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EMPTY_EXTRACTION_PREFILL", "1")
    llm = StubLLM(["", "not json at all"])
    assert _extract(llm) == {}
    assert llm.calls == [None, "{"]
