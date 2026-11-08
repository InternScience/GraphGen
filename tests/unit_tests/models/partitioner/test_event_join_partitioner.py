"""Tests for the event join partitioner."""

from __future__ import annotations

from graphgen.models.partitioner.event_join_partitioner import EventJoinPartitioner


class FakeStorage:
    def __init__(self, nodes, neighbors):
        self._nodes = nodes
        self._neighbors = neighbors

    def get_all_nodes(self):
        return list(self._nodes.items())

    def get_neighbors(self, node_id):
        return self._neighbors.get(node_id, [])


def _nodes():
    return {
        "event:anchored-1": {"entity_type": "EVENT", "graph_role": "event", "anchor": "CASE-1", "length": 1},
        "event:anchored-2": {"entity_type": "EVENT", "graph_role": "event", "anchor": "CASE-1", "length": 1},
        "event:unanchored": {"entity_type": "EVENT", "graph_role": "event", "anchor": "", "length": 1},
        "entity:operator": {"entity_type": "ACTOR", "length": 1},
    }


def test_event_join_skips_missing_anchor():
    storage = FakeStorage(
        _nodes(),
        {
            "event:anchored-1": ["entity:operator"],
            "event:anchored-2": ["entity:operator"],
            "event:unanchored": ["entity:operator"],
        },
    )

    communities = list(EventJoinPartitioner().partition(storage, require_same_anchor=True))

    assert len(communities) == 1
    assert communities[0].id == "event:anchored-1"
    assert "event:unanchored" not in communities[0].nodes
    assert set(communities[0].nodes) == {"event:anchored-1", "event:anchored-2", "entity:operator"}


def test_event_join_rejects_cross_anchor_events():
    nodes = _nodes()
    nodes["event:anchored-2"]["anchor"] = "CASE-2"
    storage = FakeStorage(
        nodes,
        {
            "event:anchored-1": ["entity:operator"],
            "event:anchored-2": ["entity:operator"],
        },
    )

    communities = list(EventJoinPartitioner().partition(storage, require_same_anchor=True))

    assert len(communities) == 2
    joined = {community.id: set(community.nodes) for community in communities}
    assert joined["event:anchored-1"] == {"event:anchored-1", "entity:operator"}
    assert joined["event:anchored-2"] == {"event:anchored-2", "entity:operator"}


def test_event_join_keeps_unanchored_events_without_anchor_requirement():
    storage = FakeStorage(
        _nodes(),
        {
            "event:unanchored": ["entity:operator"],
            "event:anchored-1": ["entity:operator"],
            "event:anchored-2": ["entity:operator"],
        },
    )

    communities = list(EventJoinPartitioner().partition(storage, require_same_anchor=False))

    assert len(communities) == 1
    assert set(communities[0].nodes) == {
        "event:anchored-1",
        "event:anchored-2",
        "event:unanchored",
        "entity:operator",
    }


def test_event_join_default_does_not_require_an_anchor():
    """The shipped generic profile stores no anchor; defaults must still partition."""
    storage = FakeStorage(
        _nodes(),
        {
            "event:unanchored": ["entity:operator"],
            "event:anchored-1": ["entity:operator"],
            "event:anchored-2": ["entity:operator"],
        },
    )

    communities = list(EventJoinPartitioner().partition(storage))

    assert len(communities) == 1
    assert "event:unanchored" in communities[0].nodes
