import random
from collections import deque
from typing import Any, Iterable, Set

from graphgen.bases import BaseGraphStorage, BasePartitioner
from graphgen.bases.datatypes import Community

random.seed(42)


class QuintuplePartitioner(BasePartitioner):
    """
    quintuple Partitioner that partitions the graph into multiple distinct quintuple (node, edge, node, edge, node).
    1. Automatically ignore isolated points.
    2. In each connected component, yield quintuples in the order of BFS.
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

                # collect all neighbors connected to node u via unused edges
                available_neighbors = []
                for v in g.get_neighbors(u):
                    edge_key = frozenset((u, v))
                    if edge_key not in used_edges:
                        available_neighbors.append(v)

                    # standard BFS queue maintenance
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)

                random.shuffle(available_neighbors)

                # every two neighbors paired with the center node u creates one quintuple
                # Note: If available_neighbors has an odd length, the remaining edge
                # stays unused for now. It may be matched into a quintuple later
                # when its other endpoint is processed as a center node.
                for i in range(0, len(available_neighbors) // 2 * 2, 2):
                    v1 = available_neighbors[i]
                    v2 = available_neighbors[i + 1]

                    edge1 = frozenset((u, v1))
                    edge2 = frozenset((u, v2))

                    used_edges.add(edge1)
                    used_edges.add(edge2)

                    v1_s, v2_s = sorted((v1, v2))

                    yield Community(
                        id=f"{v1_s}-{u}-{v2_s}",
                        nodes=[v1_s, u, v2_s],
                        edges=[tuple(sorted((v1_s, u))), tuple(sorted((u, v2_s)))],
                    )
