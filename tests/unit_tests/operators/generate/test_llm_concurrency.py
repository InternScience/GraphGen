"""Tests for bounded generator LLM concurrency."""

import asyncio

import pytest

from graphgen.operators.generate.llm_concurrency import bound_llm_concurrency


async def _fake_generate(batch, *, tracker=None, delay=0.01):
    tracker["inflight"] += 1
    tracker["peak"] = max(tracker["peak"], tracker["inflight"])
    await asyncio.sleep(delay)
    tracker["inflight"] -= 1
    return [{"question": batch, "answer": "a"}]


@pytest.mark.parametrize("limit", [1, 3])
def test_concurrency_is_bounded(limit):
    tracker = {"inflight": 0, "peak": 0}

    async def run():
        generate = bound_llm_concurrency(
            lambda batch: _fake_generate(batch, tracker=tracker), limit
        )
        await asyncio.gather(*[generate(f"b{i}") for i in range(10)])

    asyncio.run(run())
    assert tracker["peak"] <= limit


def test_non_positive_limit_fails_closed():
    with pytest.raises(ValueError, match="llm_concurrency_must_be_positive"):
        bound_llm_concurrency(_fake_generate, 0)
