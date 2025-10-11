from typing import List

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community
from graphgen.models import (
    BFSPartitioner,
    DFSPartitioner,
    ECEPartitioner,
    LeidenPartitioner,
)
from graphgen.utils import logger


def partition_kg(
    kg_instance: BaseGraphStorage,
    partition_config: dict = None,
) -> List[Community]:
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
        partitioner = ECEPartitioner()
    elif method == "leiden":
        logger.info("Partitioning knowledge graph using Leiden method.")
        partitioner = LeidenPartitioner()
    else:
        raise ValueError(f"Unsupported partition method: {method}")

    communities = partitioner.partition(g=kg_instance, **method_params)
    logger.info(f"Partitioned the graph into {len(communities)} communities.")
    return communities
