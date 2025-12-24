import asyncio
from typing import Awaitable, Callable, List, Optional, TypeVar, Union

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
    save_interval: int = 0,
    save_callback: Optional[Callable[[List[R], int], None]] = None,
    max_concurrent: Optional[int] = None,
) -> Union[List[R], Awaitable[List[R]]]:
    """
    Run coroutines concurrently with optional periodic saving.
    This function can be used in both sync and async contexts:
    - In sync context: returns List[R] directly
    - In async context: returns Awaitable[List[R]] (use with 'await')
    :return: List of results (in sync context) or coroutine (in async context)
    """
    async def _run_all():
        if not items:
            return []
        # Use semaphore to limit concurrent tasks if max_concurrent is specified
        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent is not None and max_concurrent > 0 else None

        async def run_with_semaphore(item: T) -> R:
            """Wrapper to apply semaphore if needed."""
            if semaphore:
                async with semaphore:
                    return await coro_fn(item)
            else:
                return await coro_fn(item)

        # Create tasks with concurrency limit
        if max_concurrent is not None and max_concurrent > 0:
            # Use semaphore-controlled wrapper
            tasks = [asyncio.create_task(run_with_semaphore(it)) for it in items]
        else:
            # Original behavior: create all tasks at once
            tasks = [asyncio.create_task(coro_fn(it)) for it in items]

        completed_count = 0
        results = []
        pending_save_results = []
        pbar = tqdm_async(total=len(items), desc=desc, unit=unit)

        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
                if save_interval > 0 and save_callback is not None:
                    pending_save_results.append(result)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception("Task failed: %s", e)
                # even if failed, record it to keep results consistent with tasks
                results.append(e)

            completed_count += 1
            pbar.update(1)

            # Periodic save
            if save_interval > 0 and save_callback is not None and completed_count % save_interval == 0:
                try:
                    # Filter out exceptions before saving
                    valid_results = [res for res in pending_save_results if not isinstance(res, Exception)]
                    save_callback(valid_results, completed_count)
                    pending_save_results = []  # Clear after saving
                    logger.info("Saved intermediate results: %d/%d completed", completed_count, len(items))
                except Exception as e:
                    logger.warning("Failed to save intermediate results: %s", e)

        pbar.close()

        # Save remaining results if any
        if save_interval > 0 and save_callback is not None and pending_save_results:
            try:
                valid_results = [res for res in pending_save_results if not isinstance(res, Exception)]
                save_callback(valid_results, completed_count)
                logger.info("Saved final intermediate results: %d completed", completed_count)
            except Exception as e:
                logger.warning("Failed to save final intermediate results: %s", e)

        # filter out exceptions
        return [res for res in results if not isinstance(res, Exception)]

    # Check if we're in an async context (event loop is running)
    try:
        _ = asyncio.get_running_loop()
        # If we're in an async context, return the coroutine directly
        # The caller should use 'await run_concurrent(...)'
        return _run_all()
    except RuntimeError:
        # No running loop, we can create one and run until complete
        if not items:
            return []
        loop, created = create_event_loop()
        try:
            return loop.run_until_complete(_run_all())
        finally:
            # Only close the loop if we created it
            if created:
                loop.close()
