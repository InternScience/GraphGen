"""Event-join partitioner (PMS fork, SAG join semantics).

来源：共享实体 join 语义翻译自 Zleap-AI/SAG `modules/search/base.py` 的
entity→event→relation 两跳查询（MIT，arXiv:2606.15971）；离线物化见 Phase 2 方案文档。

事件 = EVENT 节点（id 形如 ``event:{anchor}:{hash12}``）+ 星型成员边。分区算法：
1. 种子事件按节点 id 排序（确定性）；
2. 通过共享参与者实体 join 一跳邻居事件；
3. 硬约束：只接受同 project_anchor 事件（簇项目纯度 100%）；
4. 容量上限：max_events_per_community 或事件描述 token 总和。

产出官方 ``Community``（nodes = 事件+参与者实体，edges = 星型成员边），
metadata 携带 event_ids 供出题端 support 校验。
"""

import math
from collections import deque
from typing import Any, Iterable, List, Optional, Set, Tuple

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community

EVENT_NODE_PREFIX = "event:"


class EventJoinPartitioner:
    """EVENT 星型子图 → 同项目事件簇（官方 Community 数据类型）。"""

    def partition(
        self,
        g: BaseGraphStorage,
        max_events_per_community: int = 6,
        max_tokens_per_community: int = 4096,
        min_events_per_community: int = 1,
        **kwargs: Any,
    ) -> Iterable[Community]:
        events: List[Tuple[str, dict]] = []
        node_dict: dict[str, dict] = {}
        for nid, data in g.get_all_nodes():
            node_dict[nid] = data
            if str(data.get("entity_type", "")).upper() == "EVENT":
                events.append((nid, data))
        events.sort(key=lambda item: item[0])

        # 参与者 → 事件 倒排索引（共享实体 join key）
        events_by_entity: dict[str, List[str]] = {}
        star_edges: dict[str, List[str]] = {}
        for event_id, _ in events:
            star_edges[event_id] = []
            for neighbor in g.get_neighbors(event_id):
                star_edges[event_id].append(neighbor)
                events_by_entity.setdefault(neighbor, []).append(event_id)

        used: Set[str] = set()

        def _anchor(event_id: str) -> str:
            # event:{anchor}:{hash}
            return event_id[len(EVENT_NODE_PREFIX):].rsplit(":", 1)[0]

        def _grow(seed: Tuple[str, dict]) -> Optional[Community]:
            member_events: List[str] = []
            event_set: Set[str] = set()
            member_entities: dict[str, dict] = {}
            member_edges: Set[frozenset] = set()
            token_sum = 0
            anchor = _anchor(seed[0])
            queue = deque([seed[0]])

            while queue:
                event_id = queue.popleft()
                if event_id in event_set:
                    continue
                event_data = node_dict[event_id]
                if token_sum + int(event_data.get("length", 0)) > max_tokens_per_community:
                    continue
                event_set.add(event_id)
                member_events.append(event_id)
                token_sum += int(event_data.get("length", 0))
                for entity_id in star_edges.get(event_id, []):
                    if entity_id not in member_entities:
                        entity_data = node_dict.get(entity_id)
                        if entity_data is not None:
                            member_entities[entity_id] = entity_data
                    edge_key = frozenset((event_id, entity_id))
                    member_edges.add(edge_key)
                    # 共享实体 join 一跳扩展；跨锚事件被硬约束拒绝
                    for other in events_by_entity.get(entity_id, []):
                        if other in used or other in event_set:
                            continue
                        if _anchor(other) != anchor:
                            continue
                        if token_sum + int(node_dict[other].get("length", 0)) > max_tokens_per_community:
                            continue
                        queue.append(other)
                if len(event_set) >= max_events_per_community:
                    break

            if len(member_events) < min_events_per_community:
                return None
            used.update(event_set)
            return Community(
                id=seed[0],
                nodes=member_events + sorted(member_entities),
                edges=[tuple(sorted(edge)) for edge in sorted(member_edges, key=sorted)],
                metadata={"event_ids": member_events, "anchor": anchor},
            )

        for event_id, data in events:
            if event_id in used:
                continue
            community = _grow((event_id, data))
            if community:
                yield community
