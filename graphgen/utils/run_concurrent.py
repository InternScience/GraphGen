import asyncio
from typing import Awaitable, Callable, List, Optional, TypeVar

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.utils.log import logger

T = TypeVar("T")
R = TypeVar("R")


async def run_concurrent(
    coro_fn: Callable[[T], Awaitable[R]],
    items: List[T],
    *,
    desc: str = "processing",
    unit: str = "item",
    progress_bar: Optional[gr.Progress] = None,
    save_interval: int = 0,
    save_callback: Optional[Callable[[List[R], int], None]] = None,
    max_concurrent: Optional[int] = None,
) -> List[R]:
    """
    Run coroutines concurrently with optional periodic saving.
    
    :param coro_fn: Coroutine function to run for each item
    :param items: List of items to process
    :param desc: Description for progress bar
    :param unit: Unit name for progress bar
    :param progress_bar: Optional Gradio progress bar
    :param save_interval: Number of completed tasks before calling save_callback (0 to disable)
    :param save_callback: Callback function to save intermediate results (results, completed_count)
    :param max_concurrent: Maximum number of concurrent tasks (None for unlimited, default: None)
    :return: List of results
    """
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

    if progress_bar is not None:
        progress_bar(0.0, desc=f"{desc} (0/{len(items)})")

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

        if progress_bar is not None:
            progress = completed_count / len(items)
            progress_bar(progress, desc=f"{desc} ({completed_count}/{len(items)})")

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

    if progress_bar is not None:
        progress_bar(1.0, desc=f"{desc} (completed)")

    # Save remaining results if any
    if save_interval > 0 and save_callback is not None and pending_save_results:
        try:
            valid_results = [res for res in pending_save_results if not isinstance(res, Exception)]
            save_callback(valid_results, completed_count)
            logger.info("Saved final intermediate results: %d completed", completed_count)
        except Exception as e:
            logger.warning("Failed to save final intermediate results: %s", e)

    # filter out exceptions
    results = [res for res in results if not isinstance(res, Exception)]

    return results
