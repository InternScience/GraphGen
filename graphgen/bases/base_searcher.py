import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from graphgen.utils.log import set_logger


class BaseSearcher(ABC):
    """
    Abstract base class for searching and retrieving data.
    """


    def get_logger(self):
        """Get the logger instance."""
        return self.logger
