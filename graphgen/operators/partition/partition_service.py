from typing import Any

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseTokenizer
from graphgen.models import (
    AnchorBFSPartitioner,
    BFSPartitioner,
    DFSPartitioner,
    ECEPartitioner,
    LeidenPartitioner,
)
from graphgen.utils import logger


def partition_kg(
    kg_instance: BaseGraphStorage,
    chunk_storage: BaseKVStorage,
    tokenizer: Any = BaseTokenizer,
    partition_config: dict = None,
) -> list[
    tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]]
]:
    method = partition_config["method"]
    method_params = partition_config["method_params"]
    if method == "bfs":
        logger.info("Partitioning knowledge graph using BFS method.")
        partitioner = BFSPartitioner()
    elif method == "dfs":
        logger.info("Partitioning knowledge graph using DFS method.")
        partitioner = DFSPartitioner()
    elif method == "ece":
        logger.info("Partitioning knowledge graph using ECE method.")
        # TODO： before ECE partitioning, we need to:
        # 1. 'quiz and judge' to get the comprehension loss if unit_sampling is not random
        # 2. pre-tokenize nodes and edges to get the token length
        edges = kg_instance.get_all_edges()
        nodes = kg_instance.get_all_nodes()
        await pre_tokenize(kg_instance, tokenizer, edges, nodes)
        partitioner = ECEPartitioner()
    elif method == "leiden":
        logger.info("Partitioning knowledge graph using Leiden method.")
        partitioner = LeidenPartitioner()
    elif method == "anchor_bfs":
        logger.info("Partitioning knowledge graph using Anchor BFS method.")
        partitioner = AnchorBFSPartitioner(
            anchor_type=method_params.get("anchor_type"),
            anchor_ids=set(method_params.get("anchor_ids", []))
            if method_params.get("anchor_ids")
            else None,
        )
    else:
        raise ValueError(f"Unsupported partition method: {method}")

    communities = await partitioner.partition(g=kg_instance, **method_params)
    logger.info("Partitioned the graph into %d communities.", len(communities))
    batches = await partitioner.community2batch(communities, g=kg_instance)

    batches = await attach_additional_data_to_node(batches, chunk_storage)
    return batches


def attach_additional_data_to_node(
    batches: list[
        tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ]
    ],
    chunk_storage: BaseKVStorage,
) -> list[
    tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]]
]:
    """
    Attach additional data from chunk_storage to nodes in the batches.
    :param batches:
    :param chunk_storage:
    :return:
    """
    for batch in batches:
        for node_id, node_data in batch[0]:
            await _attach_by_type(node_id, node_data, chunk_storage)
    return batches


async def _attach_by_type(
    node_id: str,
    node_data: dict,
    chunk_storage: BaseKVStorage,
) -> None:
    """
    Attach additional data to the node based on its entity type.
    """
    entity_type = (node_data.get("entity_type") or "").lower()
    if not entity_type:
        return

    source_ids = [
        sid.strip()
        for sid in node_data.get("source_id", "").split("<SEP>")
        if sid.strip()
    ]

    # Handle images
    if "image" in entity_type:
        image_chunks = [
            data
            for sid in source_ids
            if "image" in sid.lower() and (data := chunk_storage.get_by_id(sid))
        ]
        if image_chunks:
            # The generator expects a dictionary with an 'img_path' key, not a list of captions.
            # We'll use the first image chunk found for this node.
            node_data["images"] = image_chunks[0]
            logger.debug("Attached image data to node %s", node_id)


import asyncio
from typing import List, Tuple

import gradio as gr

from graphgen.bases import BaseGraphStorage, BaseTokenizer
from graphgen.utils import run_concurrent


async def pre_tokenize(
    graph_storage: BaseGraphStorage,
    tokenizer: BaseTokenizer,
    edges: List[Tuple],
    nodes: List[Tuple],
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> Tuple[List, List]:
    """为 edges/nodes 补 token-length 并回写存储，并发 1000，带进度条。"""
    sem = asyncio.Semaphore(max_concurrent)

    async def _patch_and_write(obj: Tuple, *, is_node: bool) -> Tuple:
        async with sem:
            data = obj[1] if is_node else obj[2]
            if "length" not in data:
                loop = asyncio.get_event_loop()
                data["length"] = len(
                    await loop.run_in_executor(
                        None, tokenizer.encode, data["description"]
                    )
                )
            if is_node:
                graph_storage.update_node(obj[0], obj[1])
            else:
                graph_storage.update_edge(obj[0], obj[1], obj[2])
            return obj

    new_edges, new_nodes = await asyncio.gather(
        run_concurrent(
            lambda e: _patch_and_write(e, is_node=False),
            edges,
            desc="Pre-tokenizing edges",
            unit="edge",
            progress_bar=progress_bar,
        ),
        run_concurrent(
            lambda n: _patch_and_write(n, is_node=True),
            nodes,
            desc="Pre-tokenizing nodes",
            unit="node",
            progress_bar=progress_bar,
        ),
    )

    graph_storage.index_done_callback()
    return new_edges, new_nodes

