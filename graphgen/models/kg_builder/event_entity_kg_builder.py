"""Profile-driven SAG-adapted event/entity graph builder.

Domain policy (prompt, entity types, anchor rules and grounding) comes from a
validated event extraction profile. Generic graph construction remains shared.
SAG-derived adaptation and license notes are recorded in docs/PMS_PATCHES.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, Chunk
from graphgen.models.kg_builder.event_contract import event_node_name, parse_extraction_response
from graphgen.templates.event_profile_loader import EventExtractionProfile, load_event_profile
from graphgen.utils import logger

_EVENT_DESC_CAP = 4000
_ENTITY_DESC_CAP = 2000


class EventEntityKGBuilder:
    """事件-实体抽取器：extract → 星型编码节点/边；合并复用官方 LightRAG merge。"""

    def __init__(self, llm_client: BaseLLMWrapper, profile: str | EventExtractionProfile | None = None):
        from graphgen.models import LightRAGKGBuilder

        self.llm_client = llm_client
        self.profile = load_event_profile(profile) if isinstance(profile, (str, type(None))) else profile
        self._merger = LightRAGKGBuilder(llm_client=llm_client)

    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """单 chunk 事件抽取，返回与官方同形的 (nodes_data, edges_data)。"""
        content = chunk.content
        entity_types_json = json.dumps(list(self.profile.entity_types), ensure_ascii=False)
        request_json = json.dumps(
            {
                "type": "request",
                "data": {
                    "items": [{"id": 1, "content": content}],
                    "meta": {"entity_types": "__ENTITY_TYPES__"},
                },
            },
            ensure_ascii=False,
        ).replace("__ENTITY_TYPES__", entity_types_json)
        prompt = self.profile.prompt.replace("{input_text}", request_json)

        response = await self.llm_client.generate_answer(prompt)
        events, dropped = parse_extraction_response(
            response,
            chunk_content=content,
            entity_types=list(self.profile.entity_types),
            anchor_pattern=self.profile.anchor_pattern,
            anchor_required=self.profile.anchor_required,
            number_grounding=self.profile.number_grounding,
        )
        if sum(dropped.values()):
            logger.warning(
                "Event extraction dropped: %s (chunk=%s)", dropped, chunk.id
            )
        if not events:
            return {}, {}

        nodes: Dict[str, List[dict]] = defaultdict(list)
        edges: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for event in events:
            anchor = event["anchor"]
            event_name = event_node_name(
                anchor, event["title"], event["content"], legacy=self.profile.legacy_event_id
            )
            description = f"{event['title']}：{event['content']}"
            # 事件节点使用通用节点属性；分区器读取 anchor，不解析节点 ID。
            nodes[event_name].append(
                {
                    "entity_type": self.profile.event_entity_type,
                    "entity_name": event_name,
                    "graph_role": "event",
                    "description": description,
                    "anchor": anchor or "",
                    "source_id": chunk.id,
                }
            )
            # Add the configured anchor entity only when the profile requests one.
            seen: set[str] = set()
            participants: list[dict] = []
            for entity in event["entities"]:
                if entity["name"] not in seen:
                    seen.add(entity["name"])
                    participants.append(entity)
            if anchor is not None and self.profile.anchor_entity_type and anchor not in seen:
                participants.append(
                    {
                        "type": self.profile.anchor_entity_type,
                        "name": anchor,
                        "description": f"锚点 {anchor}，本事项的分组标识",
                    }
                )
                seen.add(anchor)
            for entity in participants:
                nodes[entity["name"]].append(
                    {
                        "entity_type": entity["type"],
                        "entity_name": entity["name"],
                        "description": entity["description"],
                        "source_id": chunk.id,
                    }
                )
                edges[(event_name, entity["name"])].append(
                    {
                        "src_id": event_name,
                        "tgt_id": entity["name"],
                        "description": f"事件参与者：{entity['name']}（{entity['type']}）—— {entity['description']}",
                        "source_id": chunk.id,
                        "role": entity["type"],
                    }
                )
        return dict(nodes), dict(edges)

    async def merge_nodes(self, node_data: tuple, kg_instance: BaseGraphStorage) -> dict:
        entity_name, node_list = node_data
        if node_list and str(node_list[0].get("entity_type", "")).upper() == self.profile.event_entity_type:
            return await self._merge_event_node(entity_name, node_list, kg_instance)
        # 星型事件参与者是图投影，不需要再次 LLM 摘要；只做确定性去重合并，
        # 避免 LightRAG summarizer 把跨事件上下文扩写成伪业务事实。
        return await self._merge_entity_node(entity_name, node_list, kg_instance)

    async def _merge_entity_node(
        self, entity_name: str, node_list: list, kg_instance: BaseGraphStorage
    ) -> dict:
        existing = kg_instance.get_node(entity_name) or {}
        all_rows = list(node_list)
        if existing:
            all_rows.append(existing)
        types = [str(dp.get("entity_type") or "KEYWORD") for dp in all_rows]
        entity_type = sorted(set(types), key=lambda t: (-types.count(t), t))[0]
        descriptions = sorted({str(dp.get("description") or "") for dp in all_rows if dp.get("description")})
        merged = "<SEP>".join(descriptions)[:_ENTITY_DESC_CAP]
        source_ids = sorted({sid for dp in all_rows for sid in str(dp.get("source_id", "")).split("<SEP>") if sid})
        payload = {
            "entity_type": entity_type,
            "entity_name": entity_name,
            "description": merged,
            "source_id": "<SEP>".join(source_ids),
            "length": self._merger.tokenizer.count_tokens(merged),
        }
        kg_instance.upsert_node(entity_name, node_data=payload)
        return payload

    async def _merge_event_node(
        self, entity_name: str, node_list: list, kg_instance: BaseGraphStorage
    ) -> dict:
        """事件节点确定性合并：SEP 去重拼接（截断），不走 LLM 摘要——
        官方摘要器会把事件 id 当实体名写进描述，污染出题上下文。"""
        existing = kg_instance.get_node(entity_name) or {}
        anchors = {str(dp.get("anchor") or "") for dp in node_list if dp.get("anchor")}
        existing_anchor = str(existing.get("anchor") or "")
        if existing_anchor:
            anchors.add(existing_anchor)
        merged = "<SEP>".join(sorted({dp["description"] for dp in node_list}))
        if len(merged) > _EVENT_DESC_CAP:
            merged = merged[:_EVENT_DESC_CAP]
        node_data_dict = {
            "entity_type": self.profile.event_entity_type,
            "entity_name": entity_name,
            "graph_role": "event",
            "description": merged,
            "anchor": "<SEP>".join(sorted(anchors)),
            "source_id": "<SEP>".join(
                sorted({sid for dp in node_list for sid in str(dp.get("source_id", "")).split("<SEP>") if sid}
                       | {sid for sid in str((kg_instance.get_node(entity_name) or {}).get("source_id", "")).split("<SEP>") if sid})
            ),
            "length": self._merger.tokenizer.count_tokens(merged),
        }
        kg_instance.upsert_node(entity_name, node_data=node_data_dict)
        return node_data_dict

    async def merge_edges(self, edges_data: tuple, kg_instance: BaseGraphStorage) -> dict:
        (src_id, tgt_id), edge_list = edges_data
        event_ids = {
            str(dp["src_id"])
            for dp in edge_list
            if dp.get("event_member") and dp.get("src_id")
        }
        if src_id in event_ids or tgt_id in event_ids:
            # 事件成员边同样确定性合并，不做 LLM 摘要。
            event_id = src_id if src_id in event_ids else tgt_id
            participant_id = tgt_id if event_id == src_id else src_id
            merged = "<SEP>".join(sorted({dp["description"] for dp in edge_list}))
            if len(merged) > _EVENT_DESC_CAP:
                merged = merged[:_EVENT_DESC_CAP]
            existing_edge = kg_instance.get_edge(src_id, tgt_id) or kg_instance.get_edge(tgt_id, src_id) or {}
            source_ids = {
                sid for dp in edge_list for sid in str(dp.get("source_id", "")).split("<SEP>") if sid
            } | {
                sid for sid in str(existing_edge.get("source_id", "")).split("<SEP>") if sid
            }
            edge_data = {
                "src_id": event_id,
                "tgt_id": participant_id,
                "description": merged,
                "source_id": "<SEP>".join(sorted(source_ids)),
                "length": self._merger.tokenizer.count_tokens(merged),
            }
            kg_instance.upsert_edge(event_id, participant_id, edge_data=edge_data)
            return edge_data
        return await self._merger.merge_edges(edges_data, kg_instance=kg_instance)
