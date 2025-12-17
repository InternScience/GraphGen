import asyncio
from typing import Tuple

from .log import logger


def create_event_loop() -> Tuple[asyncio.AbstractEventLoop, bool]:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        Tuple[asyncio.AbstractEventLoop, bool]: The event loop and a flag
        indicating if we created it (True) or it was already running (False).
    """
    try:
        # Try to get the running event loop (Python 3.7+)
        running_loop = asyncio.get_running_loop()
        # If we get here, there's already a running loop
        return running_loop, False
    except RuntimeError:
        # No running loop, try to get the current event loop
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_closed():
                raise RuntimeError("Event loop is closed.") from None
            # Loop exists but not running, we can use it
            return current_loop, False
        except RuntimeError:
            # No event loop exists, create a new one
            logger.info("Creating a new event loop in main thread.")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop, True
