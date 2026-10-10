"""Construct an event/entity star graph from text chunks."""

from __future__ import annotations

from collections import defaultdict
from typing import List

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models.kg_builder.event_entity_kg_builder import EventEntityKGBuilder
from graphgen.utils import run_concurrent


def build_event_entity_kg(
    llm_client: BaseLLMWrapper,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    event_profile: str | None = None,
) -> tuple:
    """chunks → 事件星型图（合并/摘要/SEP 血缘全部复用官方 LightRAG merge）。"""
    kg_builder = EventEntityKGBuilder(llm_client=llm_client, profile=event_profile)

    results = run_concurrent(
        kg_builder.extract,
        chunks,
        desc="[2/4]Extracting events and entities from chunks",
        unit="chunk",
    )
    results = [res for res in results if res]

    nodes = defaultdict(list)
    edges = defaultdict(list)
    event_ids = {
        event_id
        for node_map, _ in results
        for event_id, rows in node_map.items()
        if rows and str(rows[0].get("entity_type", "")).upper() == kg_builder.profile.event_entity_type
    }
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            normalized_key = tuple(sorted(k))
            if any(endpoint in event_ids for endpoint in normalized_key):
                for edge in v:
                    edge["event_member"] = True
            edges[normalized_key].extend(v)

    nodes = run_concurrent(
        lambda kv: kg_builder.merge_nodes(kv, kg_instance=kg_instance),
        list(nodes.items()),
        desc="Inserting events and entities into storage",
    )

    edges = run_concurrent(
        lambda kv: kg_builder.merge_edges(kv, kg_instance=kg_instance),
        list(edges.items()),
        desc="Inserting event membership into storage",
    )

    return nodes, edges
