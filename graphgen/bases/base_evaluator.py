from abc import ABC, abstractmethod
from .datatypes import QAPair


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, pair: QAPair) -> float:
        """
        Evaluate the text and return a score.
        """
