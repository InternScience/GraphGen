from typing import Any, List, Optional

try:
    import networkx as nx
except ImportError:
    nx = None

from graphgen.bases import BaseGraphStorage, BaseKVStorage


def convert_to_networkx(graph_storage: BaseGraphStorage) -> "nx.Graph":
    """Convert graph storage to NetworkX graph."""
    if nx is None:
        raise ImportError("NetworkX is required for structural evaluation")

    G = nx.DiGraph()

    # Add nodes
    nodes = graph_storage.get_all_nodes() or []
    for node_id, node_data in nodes:
        if isinstance(node_data, dict):
            G.add_node(node_id, **node_data)
        else:
            G.add_node(node_id)

    # Add edges
    edges = graph_storage.get_all_edges() or []
    for src, dst, edge_data in edges:
        if isinstance(edge_data, dict):
            G.add_edge(src, dst, **edge_data)
        else:
            G.add_edge(src, dst)

    return G


def get_source_text(chunk_storage: BaseKVStorage, chunk_id: Optional[str] = None) -> str:
    """
    Get source text from chunk storage.

    Args:
        chunk_storage: KV storage containing chunks
        chunk_id: Optional chunk ID. If None, returns concatenated text from all chunks.

    Returns:
        Source text content
    """
    if chunk_id:
        chunk = chunk_storage.get_by_id(chunk_id)
        if chunk and isinstance(chunk, dict):
            return chunk.get("content", "")
        return ""

    # Get all chunks and concatenate
    all_chunks = chunk_storage.get_all()
    texts = []
    for chunk_data in all_chunks.values():
        if isinstance(chunk_data, dict):
            content = chunk_data.get("content", "")
            if content:
                texts.append(content)
    return "\n\n".join(texts)


def get_relevant_text(
    chunk_storage: BaseKVStorage, source_id: Optional[str] = None
) -> str:
    """Get relevant source text from chunk storage using source_id."""
    if source_id:
        # Try to get specific chunk
        chunk = chunk_storage.get_by_id(source_id)
        if chunk and isinstance(chunk, dict):
            return chunk.get("content", "")
        # If source_id contains <SEP>, try multiple chunks
        if "<SEP>" in str(source_id):
            chunk_ids = [sid.strip() for sid in str(source_id).split("<SEP>") if sid.strip()]
            texts = []
            for cid in chunk_ids:
                chunk = chunk_storage.get_by_id(cid)
                if chunk and isinstance(chunk, dict):
                    content = chunk.get("content", "")
                    if content:
                        texts.append(content)
            return "\n\n".join(texts) if texts else ""

    # Fallback to all chunks
    return get_source_text(chunk_storage)


def sample_items(items: List[Any], sample_size: int) -> List[Any]:
    """Sample items from a list."""
    import random

    if len(items) <= sample_size:
        return items
    return random.sample(items, sample_size)
