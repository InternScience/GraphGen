import random
from collections import deque
from typing import Any, Iterable, Set

from graphgen.bases import BaseGraphStorage, BasePartitioner
from graphgen.bases.datatypes import Community

random.seed(42)


class TriplePartitioner(BasePartitioner):
    """
    Triple Partitioner that partitions the graph into multiple distinct triples (node, edge, node).
    1. Automatically ignore isolated points.
    2. In each connected component, yield triples in the order of BFS.
    """

    def partition(
        self,
        g: BaseGraphStorage,
        **kwargs: Any,
    ) -> Iterable[Community]:
        nodes = [n[0] for n in g.get_all_nodes()]
        random.shuffle(nodes)

        visited_nodes: Set[str] = set()
        used_edges: Set[frozenset[str]] = set()

        for seed in nodes:
            if seed in visited_nodes:
                continue

            # start BFS in a connected component
            queue = deque([seed])
            visited_nodes.add(seed)

            while queue:
                u = queue.popleft()

                for v in g.get_neighbors(u):
                    edge_key = frozenset((u, v))

                    # if this edge has not been used, a new triple has been found
                    if edge_key not in used_edges:
                        used_edges.add(edge_key)

                        # use the edge name to ensure the uniqueness of the ID
                        u_sorted, v_sorted = sorted((u, v))
                        yield Community(
                            id=f"{u_sorted}-{v_sorted}",
                            nodes=[u_sorted, v_sorted],
                            edges=[(u_sorted, v_sorted)],
                        )

                    # continue to BFS
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)
