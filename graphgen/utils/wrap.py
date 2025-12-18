from functools import wraps
from typing import Any, Callable

from .loop import create_event_loop


def async_to_sync_method(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        loop, created = create_event_loop()
        try:
            if loop.is_running():
                raise RuntimeError(
                    "Cannot use async_to_sync_method when event loop is already running."
                )
            return loop.run_until_complete(func(self, *args, **kwargs))
        finally:
            if created:
                loop.close()

    return wrapper
