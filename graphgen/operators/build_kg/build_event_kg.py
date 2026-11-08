"""Construct an event/entity star graph from text chunks."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.models.kg_builder.event_entity_kg_builder import EventEntityKGBuilder
from graphgen.utils import run_concurrent


def build_event_entity_kg(
    llm_client: BaseLLMWrapper,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    event_profile: str,
) -> Tuple[list, list]:
    """Build an event/entity star graph; merging reuses deterministic joins."""
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
    # membership edges already carry ``event_member``/``role`` from extract()
    for node_map, edge_map in results:
        for key, value in node_map.items():
            nodes[key].extend(value)
        for key, value in edge_map.items():
            edges[tuple(sorted(key))].extend(value)

    merged_nodes = run_concurrent(
        lambda kv: kg_builder.merge_nodes(kv, kg_instance=kg_instance),
        list(nodes.items()),
        desc="Inserting events and entities into storage",
    )

    merged_edges = run_concurrent(
        lambda kv: kg_builder.merge_edges(kv, kg_instance=kg_instance),
        list(edges.items()),
        desc="Inserting event membership into storage",
    )

    return merged_nodes, merged_edges
