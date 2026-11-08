"""Profile-driven event/entity star graph builder.

Domain policy (prompt, entity types, anchor rules and grounding) comes from a
validated event extraction profile. Generic graph construction is shared with
the LightRAG builder: merging event nodes, participant nodes and membership
edges is deterministic (no extra LLM summarization) so that cross-event
context is never rewritten into synthesized facts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from graphgen.bases import BaseGraphStorage, BaseKGBuilder, BaseLLMWrapper, Chunk
from graphgen.models.kg_builder.event_contract import event_node_name, parse_extraction_response
from graphgen.templates.event_profile_loader import EventExtractionProfile, load_event_profile
from graphgen.utils import logger

EVENT_DESC_CAP = 4000
ENTITY_DESC_CAP = 2000


class EventEntityKGBuilder(BaseKGBuilder):
    """Extract events and participants into a star-shaped event/entity graph."""

    def __init__(self, llm_client: BaseLLMWrapper, profile: Union[str, EventExtractionProfile]):
        super().__init__(llm_client)
        # an event profile is always required; there is no implicit default
        if profile is None:
            raise ValueError("event_profile_required")
        self.profile = profile if isinstance(profile, EventExtractionProfile) else load_event_profile(profile)
        from graphgen.models import LightRAGKGBuilder

        self._merger = LightRAGKGBuilder(llm_client=llm_client)
        self.tokenizer = self._merger.tokenizer

    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """Extract events and participants from a single chunk.

        :param chunk: input chunk
        :return: (nodes_data, edges_data) in the same shape as LightRAGKGBuilder
        """
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
            logger.warning("Event extraction dropped: %s (chunk=%s)", dropped, chunk.id)
        if not events:
            return {}, {}

        nodes: Dict[str, List[dict]] = defaultdict(list)
        edges: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for event in events:
            anchor = event["anchor"]
            event_name = event_node_name(
                anchor, event["title"], event["content"], legacy=self.profile.legacy_event_id
            )
            description = f"{event['title']}: {event['content']}"
            # The event node carries generic node attributes; the partitioner
            # reads the anchor attribute and never parses node IDs.
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
                        "description": f"Anchor {anchor} identifying this event",
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
                        "description": (
                            f"Event participant: {entity['name']} ({entity['type']})"
                            f" - {entity['description']}"
                        ),
                        "source_id": chunk.id,
                        "role": entity["type"],
                        "event_member": True,
                    }
                )
        return dict(nodes), dict(edges)

    async def merge_nodes(self, node_data: tuple, kg_instance: BaseGraphStorage) -> dict:
        entity_name, node_list = node_data
        if node_list and str(node_list[0].get("entity_type", "")).upper() == self.profile.event_entity_type:
            return await self._merge_event_node(entity_name, node_list, kg_instance)
        # Star participants are a graph projection and need no LLM summary;
        # merge deterministically to keep the LightRAG summarizer from
        # expanding cross-event context into synthesized facts.
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
        merged = "<SEP>".join(descriptions)[:ENTITY_DESC_CAP]
        source_ids = sorted({sid for dp in all_rows for sid in str(dp.get("source_id", "")).split("<SEP>") if sid})
        payload = {
            "entity_type": entity_type,
            "entity_name": entity_name,
            "description": merged,
            "source_id": "<SEP>".join(source_ids),
            "length": self.tokenizer.count_tokens(merged),
        }
        kg_instance.upsert_node(entity_name, node_data=payload)
        return payload

    async def _merge_event_node(
        self, entity_name: str, node_list: list, kg_instance: BaseGraphStorage
    ) -> dict:
        """Merge event nodes deterministically (SEP-joined, capped).

        The LightRAG summarizer would treat event IDs as entity names and leak
        them into descriptions, so events never go through LLM summarization.
        """
        existing = kg_instance.get_node(entity_name) or {}
        anchors = {str(dp.get("anchor") or "") for dp in node_list if dp.get("anchor")}
        existing_anchor = str(existing.get("anchor") or "")
        if existing_anchor:
            anchors.add(existing_anchor)
        merged = "<SEP>".join(sorted({dp["description"] for dp in node_list}))
        if len(merged) > EVENT_DESC_CAP:
            merged = merged[:EVENT_DESC_CAP]
        existing_source_ids = {
            sid
            for sid in str(existing.get("source_id", "")).split("<SEP>")
            if sid
        }
        node_data_dict = {
            "entity_type": self.profile.event_entity_type,
            "entity_name": entity_name,
            "graph_role": "event",
            "description": merged,
            "anchor": "<SEP>".join(sorted(anchors)),
            "source_id": "<SEP>".join(
                sorted(
                    existing_source_ids
                    | {sid for dp in node_list for sid in str(dp.get("source_id", "")).split("<SEP>") if sid}
                )
            ),
            "length": self.tokenizer.count_tokens(merged),
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
            # Event membership edges are merged deterministically as well.
            event_id = src_id if src_id in event_ids else tgt_id
            participant_id = tgt_id if event_id == src_id else src_id
            merged = "<SEP>".join(sorted({dp["description"] for dp in edge_list}))
            if len(merged) > EVENT_DESC_CAP:
                merged = merged[:EVENT_DESC_CAP]
            existing_edge = kg_instance.get_edge(src_id, tgt_id) or kg_instance.get_edge(tgt_id, src_id) or {}
            source_ids = {
                sid for dp in edge_list for sid in str(dp.get("source_id", "")).split("<SEP>") if sid
            } | {
                sid for sid in str(existing_edge.get("source_id", "")).split("<SEP>") if sid
            }
            role = sorted({str(dp["role"]) for dp in edge_list if dp.get("role")})
            edge_data = {
                "src_id": event_id,
                "tgt_id": participant_id,
                "description": merged,
                "source_id": "<SEP>".join(sorted(source_ids)),
                "length": self.tokenizer.count_tokens(merged),
            }
            if role:
                edge_data["role"] = "<SEP>".join(role)
            kg_instance.upsert_edge(event_id, participant_id, edge_data=edge_data)
            return edge_data
        return await self._merger.merge_edges(edges_data, kg_instance=kg_instance)
