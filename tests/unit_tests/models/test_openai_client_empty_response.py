"""``OpenAIClient.generate_answer`` 对畸形响应的处理（PMS fork 补丁 #11）。

思考模型把 ``max_tokens`` 全花在 ``<think>`` 里时，部分 OpenAI 兼容后端返回
``choices[0].message.content = None``；网关内容过滤偶发返回空 ``choices``。
官方实现直接 ``filter_think_tags(completion.choices[0].message.content)``，两种
情况都会抛 ``TypeError`` / ``IndexError``——这些既不是传输错误也不是格式错误，
池会判为不可重试并在**第一次尝试**就 fail-closed（跨源 failover 完全失效），
上层 ``run_concurrent`` 再把该 chunk 静默丢弃，造成数据缺口。

畸形响应必须归入"空响应"语义，走既有的 ``GRAPHGEN_EMPTY_RETRY`` 有界重试。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from graphgen.models.llm.api.openai_client import OpenAIClient


class _FakeCompletions:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


class _FakeTokenizer:
    """只为 ``generate_answer`` 的 prompt token 估算提供最小接口。"""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _client(responses: list) -> tuple[OpenAIClient, _FakeCompletions]:
    client = OpenAIClient(
        model="fake-model",
        api_key="fake-key",
        base_url="http://fake.invalid/v1",
        max_tokens=16,
        tokenizer=_FakeTokenizer(),
    )
    completions = _FakeCompletions(responses)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _completion(content, *, choices: bool = True):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    if not choices:
        return SimpleNamespace(choices=[], usage=usage)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_generate_answer_returns_empty_string_when_content_is_none(monkeypatch) -> None:
    """content=None（思考吃满 max_tokens）必须按空响应处理，不得抛 TypeError。"""
    monkeypatch.setenv("GRAPHGEN_EMPTY_RETRY", "0")
    client, completions = _client([_completion(None)])

    assert asyncio.run(client.generate_answer("prompt")) == ""
    assert completions.calls == 1


def test_generate_answer_returns_empty_string_when_choices_empty(monkeypatch) -> None:
    """空 choices（网关内容过滤）必须按空响应处理，不得抛 IndexError。"""
    monkeypatch.setenv("GRAPHGEN_EMPTY_RETRY", "0")
    client, completions = _client([_completion(None, choices=False)])

    assert asyncio.run(client.generate_answer("prompt")) == ""
    assert completions.calls == 1


def test_generate_answer_retries_none_content_then_succeeds(monkeypatch) -> None:
    """空响应重试预算内：先 None 后正常内容，应返回正常内容。"""
    monkeypatch.setenv("GRAPHGEN_EMPTY_RETRY", "2")
    client, completions = _client([_completion(None), _completion("entity|name")])

    assert asyncio.run(client.generate_answer("prompt")) == "entity|name"
    assert completions.calls == 2


def test_generate_answer_still_strips_think_tags(monkeypatch) -> None:
    """正常路径不受影响：<think> 段仍被剥离。"""
    monkeypatch.setenv("GRAPHGEN_EMPTY_RETRY", "0")
    client, _ = _client([_completion("<think>reasoning</think>answer")])

    assert asyncio.run(client.generate_answer("prompt")) == "answer"
