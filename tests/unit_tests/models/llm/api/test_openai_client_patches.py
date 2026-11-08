"""Tests for env-gated OpenAI client patches (retry / prefill / thinking)."""

import asyncio

from graphgen.models.llm.api.openai_client import OpenAIClient


def _client(monkeypatch, **overrides):
    monkeypatch.setenv("TOKENIZER_MODEL", "cl100k_base")
    client = OpenAIClient.__new__(OpenAIClient)
    for key, value in {
        "model": "fake-model",
        "api_key": "k",
        "api_version": None,
        "base_url": None,
        "json_mode": False,
        "seed": None,
        "topk_per_token": 5,
        "request_limit": False,
        "system_prompt": None,
    }.items():
        setattr(client, key, value)
    for key, value in overrides.items():
        setattr(client, key, value)

    class _Tok:
        def encode(self, text):
            return list(text)

    client.tokenizer = _Tok()
    client.temperature = 0
    client.top_p = 1
    client.max_tokens = 16
    client.token_usage = []

    class _RPM:
        async def wait(self, silent=True):
            return None

    class _TPM:
        async def wait(self, tokens, silent=True):
            return None

    client.rpm = _RPM()
    client.tpm = _TPM()
    return client


def _stub_completions(client, contents):
    calls = {"count": 0}

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class _Msg:
        content = None

    class _Logprobs:
        content = []

    class _Choice:
        message = _Msg()
        logprobs = _Logprobs()

    class _Completion:
        usage = _Usage()
        choices = [_Choice()]

    async def _create(**kwargs):
        idx = min(calls["count"], len(contents) - 1)
        _Choice.message.content = contents[idx]
        calls["count"] += 1
        calls.setdefault("kwargs", []).append(kwargs)
        return _Completion()

    class _StubCompletions:
        create = staticmethod(_create)

    class _StubChat:
        completions = _StubCompletions()

    class _StubClient:
        chat = _StubChat()

    client.client = _StubClient()
    return calls


def test_empty_response_retry(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EMPTY_RETRY", "2")
    client = _client(monkeypatch)
    calls = _stub_completions(client, ["", "", "real answer"])
    result = asyncio.run(client.generate_answer("hello"))
    assert result == "real answer"
    assert calls["count"] == 3


def test_empty_response_no_retry_by_default(monkeypatch):
    monkeypatch.delenv("GRAPHGEN_EMPTY_RETRY", raising=False)
    client = _client(monkeypatch)
    calls = _stub_completions(client, ["", "unused"])
    result = asyncio.run(client.generate_answer("hello"))
    assert result == ""
    assert calls["count"] == 1


def test_assistant_prefill_added_last(monkeypatch):
    client = _client(monkeypatch)
    calls = _stub_completions(client, ["ok"])
    asyncio.run(client.generate_answer("hello", assistant_prefill="{"))
    messages = calls["kwargs"][0]["messages"]
    assert messages[-1] == {"role": "assistant", "content": "{"}


def test_disable_thinking_env_gates_extra_body(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_DISABLE_THINKING", "1")
    client = _client(monkeypatch)
    calls = _stub_completions(client, ["yes"])
    asyncio.run(client.generate_topk_per_token("hello"))
    extra_body = calls["kwargs"][0]["extra_body"]
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}

    monkeypatch.delenv("GRAPHGEN_DISABLE_THINKING", raising=False)
    calls = _stub_completions(client, ["yes"])
    asyncio.run(client.generate_topk_per_token("hello"))
    assert "extra_body" not in calls["kwargs"][0]
