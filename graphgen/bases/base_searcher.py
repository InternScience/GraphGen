from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseSearcher(ABC):
    """
    Abstract base class for searching and retrieving data.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Search for data based on the given query.

        :param query: The searcher query.
        :param kwargs: Additional keyword arguments for the searcher.
        :return: Dictionary containing the searcher result, or None if not found.
        """
