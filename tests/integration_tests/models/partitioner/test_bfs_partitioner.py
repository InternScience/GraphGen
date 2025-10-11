import tempfile

import pytest

from graphgen.bases.datatypes import Community
from graphgen.models import BFSPartitioner, NetworkXStorage


@pytest.mark.asyncio
async def test_empty_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="empty")
        partitioner = BFSPartitioner()
        communities = await partitioner.partition(storage, max_units_per_community=5)
        assert communities == []


@pytest.mark.asyncio
async def test_single_node():
    nodes = [("A", {"desc": "alone"})]
    edges = []
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="single_node")

        for nid, ndata in nodes:
            await storage.upsert_node(nid, ndata)
        for src, tgt, edata in edges:
            await storage.upsert_edge(src, tgt, edata)

        partitioner = BFSPartitioner()
        communities: list[Community] = await partitioner.partition(
            storage, max_units_per_community=5
        )
        assert len(communities) == 1
        assert communities[0].nodes == ["A"]
        assert communities[0].edges == []


@pytest.mark.asyncio
async def test_small_graph():
    """
    0 - 1 - 2
    |   |   |
    3 - 4 - 5
    6 nodes & 7 edges, max_units=4 => at least 3 communities
    """
    nodes = [(str(i), {"desc": f"node{i}"}) for i in range(6)]
    edges = [
        ("0", "1", {"desc": "e01"}),
        ("1", "2", {"desc": "e12"}),
        ("0", "3", {"desc": "e03"}),
        ("1", "4", {"desc": "e14"}),
        ("2", "5", {"desc": "e25"}),
        ("3", "4", {"desc": "e34"}),
        ("4", "5", {"desc": "e45"}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="small_graph")

        for nid, ndata in nodes:
            await storage.upsert_node(nid, ndata)
        for src, tgt, edata in edges:
            await storage.upsert_edge(src, tgt, edata)

        partitioner = BFSPartitioner()
        communities: list[Community] = await partitioner.partition(
            storage, max_units_per_community=4
        )

        assert len(communities) <= 5

        all_nodes = set()
        all_edges = set()
        for c in communities:
            assert len(c.nodes) + len(c.edges) <= 4
            all_nodes.update(c.nodes)
            all_edges.update(c.edges)

        assert all_nodes == {str(i) for i in range(6)}
        assert len(all_edges) == 7
