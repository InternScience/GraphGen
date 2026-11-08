"""Tests for the event/entity star graph builder."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from graphgen.bases.datatypes import Chunk
from graphgen.models.kg_builder.event_entity_kg_builder import EventEntityKGBuilder


class FakeLLM:
    def __init__(self, response: str = ""):
        self.response = response
        self.calls = []
        self.tokenizer = SimpleNamespace(count_tokens=lambda text: len(text))

    async def generate_answer(self, prompt, **kwargs):
        self.calls.append(prompt)
        return self.response


class FakeStorage:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_edge(self, src_id, tgt_id):
        return self.edges.get((src_id, tgt_id)) or self.edges.get((tgt_id, src_id))

    def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = node_data

    def upsert_edge(self, src_id, tgt_id, edge_data):
        self.edges[(src_id, tgt_id)] = edge_data


def _response(events):
    return json.dumps({"type": "response", "data": {"items": events}}, ensure_ascii=False)


def test_extract_sets_event_member_marker_and_role():
    llm = FakeLLM(
        _response(
            [
                {
                    "title": "Server migration completed",
                    "content": "The team completed the server migration.",
                    "entities": [
                        {"type": "ACTOR", "name": "Ops Team", "description": "executed the migration"}
                    ],
                    "is_valid": True,
                }
            ]
        )
    )
    builder = EventEntityKGBuilder(llm_client=llm, profile="generic_event.v1")
    chunk = Chunk(id="chunk-1", content="The team completed the server migration.", type="text")

    nodes, edges = asyncio.run(builder.extract(chunk))

    event_ids = [nid for nid, rows in nodes.items() if rows[0].get("graph_role") == "event"]
    assert len(event_ids) == 1
    event_id = event_ids[0]
    assert nodes[event_id][0]["entity_type"] == "EVENT"
    assert nodes["Ops Team"][0]["entity_type"] == "ACTOR"

    assert len(edges) == 1
    (src, tgt), edge_rows = next(iter(edges.items()))
    assert (src, tgt) == (event_id, "Ops Team")
    assert edge_rows[0]["event_member"] is True
    assert edge_rows[0]["role"] == "ACTOR"


def test_merge_edges_persists_role_through_deterministic_merge():
    builder = EventEntityKGBuilder(llm_client=FakeLLM(), profile="generic_event.v1")
    storage = FakeStorage()
    edge = {
        "src_id": "event:case-1",
        "tgt_id": "Ops Team",
        "description": "Event participant: Ops Team (ACTOR) - executed the migration",
        "source_id": "chunk-1",
        "role": "ACTOR",
        "event_member": True,
    }

    merged = asyncio.run(builder.merge_edges((("event:case-1", "Ops Team"), [edge]), storage))

    assert merged["role"] == "ACTOR"
    assert storage.edges[("event:case-1", "Ops Team")]["role"] == "ACTOR"
    assert storage.edges[("event:case-1", "Ops Team")]["src_id"] == "event:case-1"
    assert storage.edges[("event:case-1", "Ops Team")]["tgt_id"] == "Ops Team"


def test_merge_nodes_preserves_event_role_and_anchor():
    builder = EventEntityKGBuilder(llm_client=FakeLLM(), profile="generic_event.v1")
    storage = FakeStorage()
    node_rows = [
        {
            "entity_type": "EVENT",
            "entity_name": "event:case-1",
            "graph_role": "event",
            "description": "Server migration completed: fact one",
            "anchor": "CASE-1",
            "source_id": "chunk-1",
        },
        {
            "entity_type": "EVENT",
            "entity_name": "event:case-1",
            "graph_role": "event",
            "description": "Server migration completed: fact two",
            "anchor": "CASE-1",
            "source_id": "chunk-2",
        },
    ]

    merged = asyncio.run(builder.merge_nodes(("event:case-1", node_rows), storage))

    assert merged["graph_role"] == "event"
    assert merged["anchor"] == "CASE-1"
    assert "fact one" in merged["description"] and "fact two" in merged["description"]
    assert merged["source_id"] == "chunk-1<SEP>chunk-2"
