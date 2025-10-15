from typing import Any

from graphgen.bases import BaseGraphStorage, BaseTokenizer
from graphgen.models import (
    BFSPartitioner,
    DFSPartitioner,
    ECEPartitioner,
    LeidenPartitioner,
)
from graphgen.utils import logger

from .pre_tokenize import pre_tokenize


async def partition_kg(
    kg_instance: BaseGraphStorage,
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
        edges = await kg_instance.get_all_edges()
        nodes = await kg_instance.get_all_nodes()
        await pre_tokenize(kg_instance, tokenizer, edges, nodes)
        partitioner = ECEPartitioner()
    elif method == "leiden":
        logger.info("Partitioning knowledge graph using Leiden method.")
        partitioner = LeidenPartitioner()
    else:
        raise ValueError(f"Unsupported partition method: {method}")

    communities = await partitioner.partition(g=kg_instance, **method_params)
    logger.info("Partitioned the graph into %d communities.", len(communities))
    batches = await partitioner.community2batch(communities, g=kg_instance)
    return batches
