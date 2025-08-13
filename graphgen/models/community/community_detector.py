from dataclasses import dataclass
from typing import Any, Dict

from graphgen.models.storage.networkx_storage import NetworkXStorage


@dataclass
class CommunityDetector:
    """Class for community detection algorithms."""

    graph_storage: NetworkXStorage = None
    method: str = "leiden"
    method_params: Dict[str, Any] = None

    async def detect_communities(self) -> Dict[str, int]:
        """
        Detect communities based on the chosen method.
        """
        if self.method == "leiden":
            return await self._leiden_communities(**self.method_params or {})
        raise ValueError(f"Unknown community detection method: {self.method}")

    async def get_graph(self):
        """
        Asynchronously get the graph from the storage.
        """
        return await self.graph_storage.get_graph()

    async def _leiden_communities(self, **kwargs) -> Dict[str, int]:
        """
        Detect communities using the Leiden algorithm.
        """
        import igraph as ig
        import networkx as nx
        from leidenalg import ModularityVertexPartition, find_partition

        graph = await self.get_graph()
        # Filter out isolated nodes
        graph.remove_nodes_from(list(nx.isolates(graph)))

        # Convert NetworkX graph to igraph graph
        ig_graph = ig.Graph.TupleList(graph.edges(), directed=False)

        random_seed = kwargs.get("random_seed", 42)
        use_lcc = kwargs.get("use_lcc", False)

        communities = {}
        if use_lcc:
            # Use the largest connected component
            lcc = ig_graph.components().giant()
            partition = find_partition(lcc, ModularityVertexPartition, seed=random_seed)
            for part, cluster in enumerate(partition):
                for v in cluster:
                    communities[v] = part
        else:
            offset = 0
            for component in ig_graph.components():
                subgraph = ig_graph.induced_subgraph(component)
                partition = find_partition(
                    subgraph, ModularityVertexPartition, seed=random_seed
                )
                for part, cluster in enumerate(partition):
                    for v in cluster:
                        original_node = subgraph.vs[v]["name"]
                        communities[original_node] = part + offset
                offset += len(partition)
        return communities
