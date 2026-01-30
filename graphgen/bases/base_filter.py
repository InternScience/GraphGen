from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np


class BaseFilter(ABC):
    @abstractmethod
    def filter(self, data: Any) -> bool:
        """
        Filter the data and return True if it passes the filter, False otherwise.
        """
        raise NotImplementedError


class BaseValueFilter(BaseFilter, ABC):
    @abstractmethod
    def filter(self, data: Union[int, float, np.number]) -> bool:
        """
        Filter the numeric value and return True if it passes the filter, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def filter_type(self) -> str:
        """
        Return the type of filter (e.g., "greater_than", "less_than", etc.)
        """
        raise NotImplementedError
