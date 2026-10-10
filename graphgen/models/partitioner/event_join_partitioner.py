"""Generic shared-participant event join partitioner (SAG join semantics).

Anchor isolation reads an explicit node attribute and never infers domain data
from event ID strings.
"""

from collections import deque
from typing import Any, Iterable, List, Optional, Set, Tuple

from graphgen.bases import BaseGraphStorage, BasePartitioner
from graphgen.bases.datatypes import Community


class EventJoinPartitioner(BasePartitioner):
    """EVENT 星型子图 → 同项目事件簇（官方 Community 数据类型）。"""

    def partition(
        self,
        g: BaseGraphStorage,
        max_events_per_community: int = 6,
        max_tokens_per_community: int = 4096,
        min_events_per_community: int = 1,
        event_entity_type: str = "EVENT",
        anchor_attribute: str = "anchor",
        require_same_anchor: bool = True,
        **kwargs: Any,
    ) -> Iterable[Community]:
        if not isinstance(require_same_anchor, bool):
            raise ValueError("event_join_require_same_anchor_must_be_bool")
        if not isinstance(event_entity_type, str) or not event_entity_type.strip():
            raise ValueError("event_join_event_entity_type_required")
        if not isinstance(anchor_attribute, str) or not anchor_attribute.strip():
            raise ValueError("event_join_anchor_attribute_required")
        if max_events_per_community < 1 or min_events_per_community < 1 or max_tokens_per_community < 1:
            raise ValueError("event_join_limits_must_be_positive")
        events: List[Tuple[str, dict]] = []
        node_dict: dict[str, dict] = {}
        for nid, data in g.get_all_nodes():
            node_dict[nid] = data
            if str(data.get("entity_type", "")).upper() == event_entity_type.upper():
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
            return str(node_dict[event_id].get(anchor_attribute, ""))

        def _grow(seed: Tuple[str, dict]) -> Optional[Community]:
            member_events: List[str] = []
            event_set: Set[str] = set()
            member_entities: dict[str, dict] = {}
            member_edges: Set[frozenset] = set()
            token_sum = 0
            anchor = _anchor(seed[0])
            if require_same_anchor and not anchor:
                raise ValueError(f"event_join_anchor_missing:{seed[0]}")
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
                        if require_same_anchor and _anchor(other) != anchor:
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
            )

        for event_id, data in events:
            if event_id in used:
                continue
            community = _grow((event_id, data))
            if community:
                yield community
