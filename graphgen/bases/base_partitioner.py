from abc import ABC, abstractmethod
from typing import Any, List

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Community


class BasePartitioner(ABC):
    @abstractmethod
    def partition(
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

    @staticmethod
    def community2batch(
        comm: Community, g: BaseGraphStorage
    ) -> tuple[
        list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
    ]:
        """
        Convert communities to batches of nodes and edges.
        :param comm: Community
        :param g: Graph storage instance
        :return: List of batches, each batch is a tuple of (nodes, edges)
        """
        nodes = comm.nodes
        edges = comm.edges
        nodes_data = []
        for node in nodes:
            node_data = g.get_node(node)
            if node_data:
                nodes_data.append((node, node_data))
        edges_data = []
        for edge in edges:
            # Filter out self-loops and invalid edges
            if not isinstance(edge, tuple) or len(edge) != 2:
                continue
            u, v = edge
            if u == v:
                continue

            edge_data = g.get_edge(u, v) or g.get_edge(v, u)
            if edge_data:
                edges_data.append((u, v, edge_data))
        return nodes_data, edges_data

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
        for u, v, _ in edges:
            if u == v:
                continue
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        return adj, edge_set
