import random
from collections import deque
from typing import Any, Iterable, List, Literal, Set, Tuple, Union

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community

from .bfs_partitioner import BFSPartitioner

NODE_UNIT: str = "n"
EDGE_UNIT: str = "e"


class AnchorBFSPartitioner(BFSPartitioner):
    """
    Anchor BFS partitioner that partitions the graph into communities of a fixed size.
    1. Randomly choose a node of a specified type as the anchor.
    2. Expand the community using BFS until the max unit size is reached.(A unit is a node or an edge.)
    3. Non-anchor units can only be "pulled" into a community and never become seeds themselves.
    For example, for VQA tasks, we may want to use image nodes as anchors and expand to nearby text nodes and edges.

    Supports multiple anchor types for multi-omics data: anchor_type can be a single string or a list of strings.
    When a list is provided, nodes matching any of the types in the list can serve as anchors.
    """

    def __init__(
        self,
        *,
        anchor_type: Union[
            Literal["image", "dna", "rna", "protein"],
            List[Literal["dna", "rna", "protein"]],
        ] = "image",
        anchor_ids: Set[str] | None = None,
    ) -> None:
        super().__init__()
        # Normalize anchor_type to always be a list for internal processing
        if isinstance(anchor_type, str):
            self.anchor_types = [anchor_type]
        else:
            self.anchor_types = list(anchor_type)
        self.anchor_ids = anchor_ids

    def partition(
        self,
        g: BaseGraphStorage,
        max_units_per_community: int = 1,
        **kwargs: Any,
    ) -> Iterable[Community]:
        nodes = g.get_all_nodes()  # List[tuple[id, meta]]
        edges = g.get_all_edges()  # List[tuple[u, v, meta]]

        adj, _ = self._build_adjacency_list(nodes, edges)

        anchors: Set[str] = self._pick_anchor_ids(nodes)
        if not anchors:
            return  # if no anchors, return nothing

        used_n: set[str] = set()
        used_e: set[frozenset[str]] = set()

        seeds = list(anchors)
        random.shuffle(seeds)

        for seed_node in seeds:
            if seed_node in used_n:
                continue
            comm_n, comm_e = self._grow_community(
                seed_node, adj, max_units_per_community, used_n, used_e
            )
            if comm_n or comm_e:
                yield Community(id=seed_node, nodes=comm_n, edges=comm_e)

    def _pick_anchor_ids(
        self,
        nodes: List[tuple[str, dict]],
    ) -> Set[str]:
        if self.anchor_ids is not None:
            return self.anchor_ids

        anchor_ids: Set[str] = set()
        anchor_types_lower = [at.lower() for at in self.anchor_types]

        for node_id, meta in nodes:
            # Check if node matches any of the anchor types
            matched = False

            # Check 1: entity_type (for image, etc.)
            node_type = str(meta.get("entity_type", "")).lower()
            for anchor_type_lower in anchor_types_lower:
                if anchor_type_lower in node_type:
                    anchor_ids.add(node_id)
                    matched = True
                    break

            if matched:
                continue

            # Check 2: molecule_type (for omics data: dna, rna, protein)
            molecule_type = str(meta.get("molecule_type", "")).lower()
            if molecule_type in anchor_types_lower:
                anchor_ids.add(node_id)
                continue

            # Check 3: source_id prefix (for omics data: dna-, rna-, protein-)
            source_id = str(meta.get("source_id", "")).lower()
            for anchor_type_lower in anchor_types_lower:
                if source_id.startswith(f"{anchor_type_lower}-"):
                    anchor_ids.add(node_id)
                    matched = True
                    break

            if matched:
                continue

            # Check 4: Check if source_id contains multiple IDs separated by <SEP>
            if "<sep>" in source_id:
                source_ids = source_id.split("<sep>")
                for sid in source_ids:
                    sid = sid.strip()
                    for anchor_type_lower in anchor_types_lower:
                        if sid.startswith(f"{anchor_type_lower}-"):
                            anchor_ids.add(node_id)
                            matched = True
                            break
                    if matched:
                        break

        return anchor_ids

    @staticmethod
    def _grow_community(
        seed: str,
        adj: dict[str, List[str]],
        max_units: int,
        used_n: set[str],
        used_e: set[frozenset[str]],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Grow a community from the seed node using BFS.
        :param seed: seed node id
        :param adj: adjacency list
        :param max_units: maximum number of units (nodes + edges) in the community
        :param used_n: set of used node ids
        :param used_e: set of used edge keys
        :return: (list of node ids, list of edge tuples)
        """
        comm_n: List[str] = []
        comm_e: List[Tuple[str, str]] = []
        queue: deque[tuple[str, Any]] = deque([(NODE_UNIT, seed)])
        cnt = 0

        while queue and cnt < max_units:
            k, it = queue.popleft()

            if k == NODE_UNIT:
                if it in used_n:
                    continue
                used_n.add(it)
                comm_n.append(it)
                cnt += 1
                for nei in adj[it]:
                    e_key = frozenset((it, nei))
                    if e_key not in used_e:
                        queue.append((EDGE_UNIT, e_key))
            else:  # EDGE_UNIT
                if it in used_e:
                    continue
                used_e.add(it)
                # Convert frozenset to tuple for edge representation
                # Note: Self-loops should be filtered during graph construction,
                # but we handle edge cases defensively
                try:
                    u, v = tuple(it)
                except ValueError:
                    # Handle edge case: frozenset with unexpected number of elements
                    # This should not happen if graph construction is correct
                    edge_nodes = list(it)
                    if len(edge_nodes) == 1:
                        # Self-loop edge (should have been filtered during graph construction)
                        u, v = edge_nodes[0], edge_nodes[0]
                    else:
                        # Invalid edge, skip it
                        continue
                comm_e.append((u, v))
                cnt += 1
                for n in it:
                    if n not in used_n:
                        queue.append((NODE_UNIT, n))

        return comm_n, comm_e
