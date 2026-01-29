import asyncio
from typing import Awaitable, Callable, List, TypeVar

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.utils.log import logger

from .loop import create_event_loop

T = TypeVar("T")
R = TypeVar("R")


def run_concurrent(
    coro_fn: Callable[[T], Awaitable[R]],
    items: List[T],
    *,
    desc: str = "processing",
    unit: str = "item",
) -> List[R]:
    async def _run_all():
        # Wrapper to return the index alongside the result
        # This eliminates the need to map task IDs
        async def _worker(index: int, item: T):
            try:
                res = await coro_fn(item)
                return index, res, None
            except Exception as e:
                return index, None, e

        # Create tasks using the wrapper
        tasks_list = [
            asyncio.create_task(_worker(i, item)) for i, item in enumerate(items)
        ]

        results: List[Exception | R] = [None] * len(items)
        pbar = tqdm_async(total=len(items), desc=desc, unit=unit)

        # Iterate over completed tasks
        for future in asyncio.as_completed(tasks_list):
            # We await the wrapper, which guarantees we get the index back
            idx, result, error = await future

            if error:
                logger.exception(f"Task failed at index {idx}: {error}")
            else:
                results[idx] = result

            pbar.update(1)

        pbar.close()
        return results

    loop = create_event_loop()
    try:
        return loop.run_until_complete(_run_all())
    finally:
        loop.close()
