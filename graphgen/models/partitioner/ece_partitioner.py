import math
import random
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community
from graphgen.models.partitioner.bfs_partitioner import BFSPartitioner

NODE_UNIT: str = "n"
EDGE_UNIT: str = "e"


class ECEPartitioner(BFSPartitioner):
    """
    ECE partitioner that partitions the graph into communities based on Expected Calibration Error (ECE).
    We calculate ECE for units in KG (represented as 'comprehension loss')
    and group units with similar ECE values into the same community.
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
        default_loss = -math.log(0.1)
        if edge_sampling == "random":
            random.shuffle(units)
        elif edge_sampling == "min_loss":
            units = sorted(
                units,
                key=lambda x: x[-1].get("loss", default_loss),
            )
        elif edge_sampling == "max_loss":
            units = sorted(
                units,
                key=lambda x: x[-1].get("loss", default_loss),
                reverse=True,
            )
        else:
            raise ValueError(f"Invalid edge sampling: {edge_sampling}")
        return units

    def partition(
        self,
        g: BaseGraphStorage,
        max_units_per_community: int = 10,
        min_units_per_community: int = 1,
        max_tokens_per_community: int = 10240,
        unit_sampling: str = "random",
        min_loss_coverage: float = 0.95,
        **kwargs: Any,
    ) -> Iterable[Community]:
        nodes: List[Tuple[str, dict]] = g.get_all_nodes()
        edges: List[Tuple[str, str, dict]] = g.get_all_edges()

        node_dict = dict(nodes)
        edge_dict = {frozenset((u, v)): d for u, v, d in edges}

        all_units: List[Tuple[str, Any, dict]] = [
            (NODE_UNIT, nid, d) for nid, d in nodes
        ] + [(EDGE_UNIT, frozenset((u, v)), d) for u, v, d in edges]

        # PMS_PATCHES.md #13：loss 采样下，"没跑 judge / TRAINEE 漏配 / judge
        # 逐条失败"的单元都会落 default_loss≈2.30 的假高 loss，恰聚在 max_loss
        # 排序最前——run 正常出题但零信号。这里 fail-closed：真实 loss 覆盖率
        # 低于阈值直接拒绝分区。
        if unit_sampling != "random" and all_units:
            default_loss = -math.log(0.1)
            judged = [
                unit
                for unit in all_units
                if isinstance(unit[-1].get("loss"), (int, float))
                and abs(unit[-1]["loss"] - default_loss) > 1e-9
            ]
            coverage = len(judged) / len(all_units)
            if coverage < min_loss_coverage:
                raise ValueError(
                    f"partition_loss_coverage_too_low:{coverage:.3f}<{min_loss_coverage}"
                    " — loss 采样要求 quiz→judge 先产真实 loss（TRAINEE_* 配置、"
                    "judge 失败率、上游单元枚举都可能是缺口）"
                )

        used_n: Set[str] = set()
        used_e: Set[frozenset[str]] = set()

        all_units = self._sort_units(all_units, unit_sampling)

        def _grow_community(seed_unit: Tuple[str, Any, dict]) -> Optional[Community]:
            nonlocal used_n, used_e

            community_nodes: Dict[str, dict] = {}
            community_edges: Dict[frozenset[str], dict] = {}
            queue = deque()
            token_sum = 0

            def _add_unit(u):
                nonlocal token_sum
                t, i, d = u
                if t == NODE_UNIT:  # node
                    if i in used_n or i in community_nodes:
                        return False
                    community_nodes[i] = d
                    used_n.add(i)
                else:  # edge
                    if i in used_e or i in community_edges:
                        return False
                    community_edges[i] = d
                    used_e.add(i)
                token_sum += int(d.get("length", 0))
                return True

            _add_unit(seed_unit)
            queue.append(seed_unit)

            # BFS
            while queue:
                if (
                    len(community_nodes) + len(community_edges)
                    >= max_units_per_community
                    or token_sum >= max_tokens_per_community
                ):
                    break

                cur_type, cur_id, _ = queue.popleft()

                neighbors: List[Tuple[str, Any, dict]] = []
                if cur_type == NODE_UNIT:
                    for nb_id in g.get_neighbors(cur_id):
                        e_key = frozenset((cur_id, nb_id))
                        if e_key not in used_e and e_key not in community_edges:
                            neighbors.append((EDGE_UNIT, e_key, edge_dict[e_key]))
                else:
                    for n_id in cur_id:
                        if n_id not in used_n and n_id not in community_nodes:
                            neighbors.append((NODE_UNIT, n_id, node_dict[n_id]))

                neighbors = self._sort_units(neighbors, unit_sampling)
                for nb in neighbors:
                    if (
                        len(community_nodes) + len(community_edges)
                        >= max_units_per_community
                        or token_sum >= max_tokens_per_community
                    ):
                        break
                    if _add_unit(nb):
                        queue.append(nb)

            if len(community_nodes) + len(community_edges) < min_units_per_community:
                return None

            return Community(
                id=seed_unit[1],
                nodes=list(community_nodes.keys()),
                edges=[tuple(sorted(e)) for e in community_edges],
            )

        for unit in tqdm(all_units, desc="ECE partition"):
            utype, uid, _ = unit
            if (utype == NODE_UNIT and uid in used_n) or (
                utype == EDGE_UNIT and uid in used_e
            ):
                continue
            comm = _grow_community(unit)
            if comm:
                yield comm
