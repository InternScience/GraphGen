import tempfile

import pytest

from graphgen.bases.datatypes import Community
from graphgen.models import ECEPartitioner, NetworkXStorage


@pytest.mark.asyncio
async def test_ece_empty_graph():
    """ECE partitioning on an empty graph should return an empty community list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="empty")
        partitioner = ECEPartitioner()
        communities = await partitioner.partition(
            storage, max_units_per_community=5, unit_sampling="random"
        )
        assert communities == []


@pytest.mark.asyncio
async def test_ece_single_node():
    """A single node must be placed in exactly one community under any edge-sampling strategy."""
    nodes = [("A", {"desc": "alone", "length": 10, "loss": 0.1})]

    for strategy in ("random", "min_loss", "max_loss"):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = NetworkXStorage(
                working_dir=tmpdir, namespace=f"single_{strategy}"
            )
            for nid, ndata in nodes:
                await storage.upsert_node(nid, ndata)

            partitioner = ECEPartitioner()
            communities: list[Community] = await partitioner.partition(
                storage, max_units_per_community=5, unit_sampling=strategy
            )
            assert len(communities) == 1
            assert communities[0].nodes == ["A"]
            assert communities[0].edges == []


@pytest.mark.asyncio
async def test_ece_small_graph_random():
    """
    2x3 grid graph:
        0 — 1 — 2
        |   |   |
        3 — 4 — 5
    6 nodes & 7 edges, max_units=4  =>  at least 3 communities expected with random sampling.
    """
    nodes = [(str(i), {"desc": f"node{i}", "length": 10}) for i in range(6)]
    edges = [
        ("0", "1", {"desc": "e01", "loss": 0.1, "length": 5}),
        ("1", "2", {"desc": "e12", "loss": 0.2, "length": 5}),
        ("0", "3", {"desc": "e03", "loss": 0.3, "length": 5}),
        ("1", "4", {"desc": "e14", "loss": 0.4, "length": 5}),
        ("2", "5", {"desc": "e25", "loss": 0.5, "length": 5}),
        ("3", "4", {"desc": "e34", "loss": 0.6, "length": 5}),
        ("4", "5", {"desc": "e45", "loss": 0.7, "length": 5}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="small_random")
        for nid, ndata in nodes:
            await storage.upsert_node(nid, ndata)
        for src, tgt, edata in edges:
            await storage.upsert_edge(src, tgt, edata)

        partitioner = ECEPartitioner()
        communities: list[Community] = await partitioner.partition(
            storage, max_units_per_community=4, unit_sampling="random"
        )

        # Basic integrity checks
        all_nodes = set()
        all_edges = set()
        for c in communities:
            assert len(c.nodes) + len(c.edges) <= 4
            all_nodes.update(c.nodes)
            all_edges.update((u, v) if u < v else (v, u) for u, v in c.edges)
        assert all_nodes == {str(i) for i in range(6)}
        assert len(all_edges) == 7


@pytest.mark.asyncio
async def test_ece_small_graph_min_loss():
    """
    Same grid graph, but using min_loss sampling.
    Edges with lower loss should be preferred during community expansion.
    """
    nodes = [
        (str(i), {"desc": f"node{i}", "length": 10, "loss": i * 0.1}) for i in range(6)
    ]
    edges = [
        ("0", "1", {"desc": "e01", "loss": 0.05, "length": 5}),
        ("1", "2", {"desc": "e12", "loss": 0.10, "length": 5}),
        ("0", "3", {"desc": "e03", "loss": 0.15, "length": 5}),
        ("1", "4", {"desc": "e14", "loss": 0.20, "length": 5}),
        ("2", "5", {"desc": "e25", "loss": 0.25, "length": 5}),
        ("3", "4", {"desc": "e34", "loss": 0.30, "length": 5}),
        ("4", "5", {"desc": "e45", "loss": 0.35, "length": 5}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="small_min")
        for nid, ndata in nodes:
            await storage.upsert_node(nid, ndata)
        for src, tgt, edata in edges:
            await storage.upsert_edge(src, tgt, edata)

        partitioner = ECEPartitioner()
        communities: list[Community] = await partitioner.partition(
            storage, max_units_per_community=4, unit_sampling="min_loss"
        )

        all_nodes = set()
        all_edges = set()
        for c in communities:
            assert len(c.nodes) + len(c.edges) <= 4
            all_nodes.update(c.nodes)
            all_edges.update((u, v) if u < v else (v, u) for u, v in c.edges)
        assert all_nodes == {str(i) for i in range(6)}
        assert len(all_edges) == 7


@pytest.mark.asyncio
async def test_ece_small_graph_max_loss():
    """
    Same grid graph, but using max_loss sampling.
    Edges with higher loss should be preferred during community expansion.
    """
    nodes = [
        (str(i), {"desc": f"node{i}", "length": 10, "loss": (5 - i) * 0.1})
        for i in range(6)
    ]
    edges = [
        ("0", "1", {"desc": "e01", "loss": 0.35, "length": 5}),
        ("1", "2", {"desc": "e12", "loss": 0.30, "length": 5}),
        ("0", "3", {"desc": "e03", "loss": 0.25, "length": 5}),
        ("1", "4", {"desc": "e14", "loss": 0.20, "length": 5}),
        ("2", "5", {"desc": "e25", "loss": 0.15, "length": 5}),
        ("3", "4", {"desc": "e34", "loss": 0.10, "length": 5}),
        ("4", "5", {"desc": "e45", "loss": 0.05, "length": 5}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="small_max")
        for nid, ndata in nodes:
            await storage.upsert_node(nid, ndata)
        for src, tgt, edata in edges:
            await storage.upsert_edge(src, tgt, edata)

        partitioner = ECEPartitioner()
        communities: list[Community] = await partitioner.partition(
            storage, max_units_per_community=4, unit_sampling="max_loss"
        )

        all_nodes = set()
        all_edges = set()
        for c in communities:
            assert len(c.nodes) + len(c.edges) <= 4
            all_nodes.update(c.nodes)
            all_edges.update((u, v) if u < v else (v, u) for u, v in c.edges)
        assert all_nodes == {str(i) for i in range(6)}
        assert len(all_edges) == 7


@pytest.mark.asyncio
async def test_ece_max_tokens_limit():
    """Ensure max_tokens_per_community is respected."""
    # node id -> data
    node_data = {"A": {"length": 3000}, "B": {"length": 3000}, "C": {"length": 3000}}
    # edge list
    edges = [("A", "B", {"loss": 0.1, "length": 2000})]

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = NetworkXStorage(working_dir=tmpdir, namespace="token_limit")
        for nid, ndata in node_data.items():
            await storage.upsert_node(nid, ndata)
        for src, tgt, edata in edges:
            await storage.upsert_edge(src, tgt, edata)

        partitioner = ECEPartitioner()
        communities: list[Community] = await partitioner.partition(
            storage,
            max_units_per_community=10,
            max_tokens_per_community=5000,  # 1 node (3000) + 1 edge (2000) = 5000
            unit_sampling="random",
        )

        # With a 5000-token budget we need at least two communities
        assert len(communities) >= 2

        # helper: quick edge lookup
        edge_lens = {(u, v): d["length"] for u, v, d in edges}
        edge_lens.update({(v, u): d["length"] for u, v, d in edges})  # undirected

        for c in communities:
            node_tokens = sum(node_data[n]["length"] for n in c.nodes)
            edge_tokens = sum(edge_lens[e] for e in c.edges)
            assert node_tokens + edge_tokens <= 5000
