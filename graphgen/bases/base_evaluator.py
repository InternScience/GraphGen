from abc import ABC, abstractmethod
from typing import Any

from .base_storage import BaseGraphStorage
from .datatypes import QAPair


class BaseQAEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, pair: QAPair) -> dict[str, float]:
        """
        Evaluate the text and return a score.
        """


class BaseKGEvaluator(ABC):
    @abstractmethod
    def evaluate(self, kg: BaseGraphStorage) -> dict[str, Any]:
        """
        Evaluate the whole graph and return a dict of scores.
        """


class BaseTripleEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, unit: dict) -> dict[str, float]:
        """
        Evaluate a node/edge and return a score.
        """
