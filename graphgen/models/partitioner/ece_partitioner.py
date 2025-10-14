import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community
from graphgen.models.partitioner.bfs_partitioner import BFSPartitioner


@dataclass
class ECEPartitioner(BFSPartitioner):
    """
    ECE partitioner that partitions the graph into communities based on Expected Calibration Error (ECE).
    We calculate ECE for edges in KG(represented as 'comprehension loss')
    and group edges with similar ECE values into the same community.
    1. Select a sampling strategy.
    2. Choose a unit based on the sampling strategy.
    2. Expand the community using BFS.
    3. When expending, prefer to add units with the sampling strategy.
    4. Stop when the max unit size is reached or the max input length is reached.
    (A unit is a node or an edge.)
    """

    @staticmethod
    def _sort_units(units: list, edge_sampling: str) -> list:
        """
        Sort units with edge sampling strategy

        :param units: total units
        :param edge_sampling: edge sampling strategy (random, min_loss, max_loss)
        :return: sorted units
        """
        if edge_sampling == "random":
            random.shuffle(units)
        elif edge_sampling == "min_loss":
            units = sorted(
                units,
                key=lambda x: x[-1]["loss"],
            )
        elif edge_sampling == "max_loss":
            units = sorted(
                units,
                key=lambda x: x[-1]["loss"],
                reverse=True,
            )
        else:
            raise ValueError(f"Invalid edge sampling: {edge_sampling}")
        return units

    async def partition(
        self,
        g: BaseGraphStorage,
        max_units_per_community: int = 10,
        max_tokens_per_community: int = 10240,
        edge_sampling: str = "random",
        **kwargs: Any,
    ) -> List[Community]:
        nodes: List[Tuple[str, dict]] = await g.get_all_nodes()
        edges: List[Tuple[str, str, dict]] = await g.get_all_edges()

        adj, _ = self._build_adjacency_list(nodes, edges)
        node_dict = dict(nodes)
        edge_dict = {frozenset((u, v)): d for u, v, d in edges}

        all_units: List[Tuple[str, Any, dict]] = [("n", nid, d) for nid, d in nodes] + [
            ("e", frozenset((u, v)), d) for u, v, d in edges
        ]

        used_n: Set[str] = set()
        used_e: Set[frozenset[str]] = set()
        communities: List = []

        all_units = self._sort_units(all_units, edge_sampling)

        async def _grow_community(seed_unit: Tuple[str, Any, dict]) -> Community:
            nonlocal used_n, used_e

            community_nodes: Dict[str, dict] = {}
            community_edges: Dict[frozenset[str], dict] = {}
            queue: asyncio.Queue = asyncio.Queue()
            token_sum = 0

            async def _add_unit(u):
                nonlocal token_sum
                t, i, d = u
                if t == "n":
                    if i in used_n or i in community_nodes:
                        return False
                    community_nodes[i] = d
                    used_n.add(i)
                else:  # edge
                    if i in used_e or i in community_edges:
                        return False
                    community_edges[i] = d
                    used_e.add(i)
                token_sum += d.get("length", 0)
                return True

            await _add_unit(seed_unit)
            await queue.put(seed_unit)

            # BFS
            while not queue.empty():
                if (
                    len(community_nodes) + len(community_edges)
                    >= max_units_per_community
                    or token_sum >= max_tokens_per_community
                ):
                    break

                cur_type, cur_id, _ = await queue.get()

                neighbors: List[Tuple[str, Any, dict]] = []
                if cur_type == "n":
                    for nb_id in adj.get(cur_id, []):
                        e_key = frozenset((cur_id, nb_id))
                        if e_key not in used_e and e_key not in community_edges:
                            neighbors.append(("e", e_key, edge_dict[e_key]))
                else:
                    for n_id in cur_id:
                        if n_id not in used_n and n_id not in community_nodes:
                            neighbors.append(("n", n_id, node_dict[n_id]))

                neighbors = self._sort_units(neighbors, edge_sampling)
                for nb in neighbors:
                    if (
                        len(community_nodes) + len(community_edges)
                        >= max_units_per_community
                        or token_sum >= max_tokens_per_community
                    ):
                        break
                    if await _add_unit(nb):
                        await queue.put(nb)

            return Community(
                id=len(communities),
                nodes=list(community_nodes.keys()),
                edges=[(u, v) for (u, v), _ in community_edges.items()],
            )

        async for unit in tqdm_async(all_units, desc="ECE partition"):
            utype, uid, _ = unit
            if (utype == "n" and uid in used_n) or (utype == "e" and uid in used_e):
                continue
            communities.append(await _grow_community(unit))

        return communities
