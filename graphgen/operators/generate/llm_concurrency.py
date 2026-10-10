"""Bound in-flight LLM requests of a generator inside one event loop.

Official ``run_concurrent`` has no cross-batch concurrency cap; a busy LLM
endpoint (e.g. a 35B-A3B deployment) can deadlock under high fan-out. This
helper caps concurrent ``generator.generate`` calls with an asyncio.Semaphore
created lazily per running loop (Ray actors create a fresh loop per batch).
Pure stdlib so PMS-side contract tests can import this module by path without
the graphgen package installed. See docs/PMS_PATCHES.md #3.
"""

from __future__ import annotations

import asyncio
from typing import Any


def bound_llm_concurrency(generate: Any, limit: int) -> Any:
    """Wrap an async ``generate(batch)`` so at most ``limit`` run in-flight."""
    if limit < 1:
        raise ValueError("llm_concurrency_must_be_positive")
    state: dict[str, Any] = {}

    async def _gated_generate(batch: Any) -> Any:
        loop = asyncio.get_running_loop()
        semaphore = state.get("semaphore")
        if semaphore is None or state.get("loop") is not loop:
            semaphore = asyncio.Semaphore(limit)
            state["semaphore"] = semaphore
            state["loop"] = loop
        async with semaphore:
            return await generate(batch)

    return _gated_generate
