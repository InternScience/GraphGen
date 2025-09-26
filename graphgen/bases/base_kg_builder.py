from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from graphgen.bases.datatypes import Chunk


@dataclass
class BaseKGBuilder(ABC):

    # node_types: List[str]
    def build(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def extract(self, chunk: Chunk) -> None:
        pass

    # 摘要
    def condense(self) -> None:
        pass

    def _merge_nodes(self) -> None:
        pass

    def _merge_edges(self) -> None:
        pass
