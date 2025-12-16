import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any

from rich.logging import RichHandler


def set_logger(
    log_file: str,
    name: str,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    *,
    if_stream: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 5,
    force: bool = False,
):

    current_logger = logging.getLogger(name)
    if current_logger.hasHandlers() and not force:
        return current_logger

    if force:
        current_logger.handlers.clear()

    current_logger.setLevel(
        min(file_level, console_level)
    )  # Set to the lowest level to capture all logs
    current_logger.propagate = False

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if if_stream:
        console = RichHandler(
            level=console_level, show_path=False, rich_tracebacks=True
        )
        console.setFormatter(logging.Formatter("%(message)s"))
        current_logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
    )
    current_logger.addHandler(file_handler)
    return current_logger


CURRENT_LOGGER_VAR = contextvars.ContextVar("current_logger")


def get_current_logger() -> logging.Logger:
    current_logger = CURRENT_LOGGER_VAR.get()
    if not current_logger:
        raise RuntimeError("No logger is set in the current context.")
    return current_logger


class ContextAwareLogger:
    @staticmethod
    def _get_logger() -> logging.Logger:
        return get_current_logger()

    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().error(msg, *args, **kwargs)

    def exception(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().exception(msg, *args, **kwargs)

    def critical(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().critical(msg, *args, **kwargs)

    def log(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
        self._get_logger().log(level, msg, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_logger(), name)


logger = ContextAwareLogger()
