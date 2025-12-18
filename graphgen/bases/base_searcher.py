import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from graphgen.utils.log import set_logger


class BaseSearcher(ABC):
    """
    Abstract base class for searching and retrieving data.
    """

    def __init__(self, working_dir: str = "cache"):
        """
        Initialize the base searcher with a logger.

        :param working_dir: Working directory for log files.
        """
        log_dir = os.path.join(working_dir, "logs")
        searcher_name = self.__class__.__name__

        # e.g. cache/logs/NCBISearch.log
        log_file = os.path.join(log_dir, f"{searcher_name}.log")

        self.logger = set_logger(
            log_file=log_file, name=searcher_name,
            console_level=logging.ERROR, force=True
        )

        self.logger.info(
            "[%s] Searcher initialized", searcher_name
        )

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for data based on the given query.

        :param query: The searcher query.
        :param kwargs: Additional keyword arguments for the searcher.
        :return: List of dictionaries containing the searcher results.
        """

    def get_logger(self):
        """Get the logger instance."""
        return self.logger
