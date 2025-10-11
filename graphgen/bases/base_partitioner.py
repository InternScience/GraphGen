from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Community


@dataclass
class BasePartitioner(ABC):
    @abstractmethod
    async def partition(
        self,
        g: BaseGraphStorage,
        **kwargs: Any,
    ) -> List[Community]:
        """
        Graph -> Communities
        :param g: Graph storage instance
        :param kwargs: Additional parameters for partitioning
        :return: List of communities
        """

    @abstractmethod
    def split_communities(self, communities: List[Community]) -> List[Community]:
        """
        Split large communities into smaller ones based on max_size.
        :param communities
        :return:
        """

    @staticmethod
    def _build_adjacency_list(
        nodes: List[tuple[str, dict]], edges: List[tuple[str, str, dict]]
    ) -> tuple[dict[str, List[str]], set[tuple[str, str]]]:
        """
        Build adjacency list and edge set from nodes and edges.
        :param nodes
        :param edges
        :return: adjacency list, edge set
        """
        adj: dict[str, List[str]] = {n[0]: [] for n in nodes}
        edge_set: set[tuple[str, str]] = set()
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
            edge_set.add((e[0], e[1]))
            edge_set.add((e[1], e[0]))
        return adj, edge_set
