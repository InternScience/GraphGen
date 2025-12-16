import json
import os
from typing import Iterable

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseOperator, BaseTokenizer
from graphgen.common import init_storage
from graphgen.models import (
    AnchorBFSPartitioner,
    BFSPartitioner,
    DFSPartitioner,
    ECEPartitioner,
    LeidenPartitioner,
    Tokenizer,
)
from graphgen.utils import logger


class PartitionService(BaseOperator):
    def __init__(self, working_dir: str = "cache", **partition_kwargs):
        super().__init__(working_dir=working_dir, op_name="partition_service")
        self.kg_instance: BaseGraphStorage = init_storage(
            backend="kuzu",
            working_dir=working_dir,
            namespace="graph",
        )
        self.chunk_storage: BaseKVStorage = init_storage(
            backend="rocksdb",
            working_dir=working_dir,
            namespace="chunk",
        )
        tokenizer_model = os.getenv("TOKENIZER_MODEL", "cl100k_base")
        self.tokenizer_instance: BaseTokenizer = Tokenizer(model_name=tokenizer_model)
        self.partition_kwargs = partition_kwargs

    def process(self, batch: pd.DataFrame) -> Iterable[pd.DataFrame]:
        # this operator does not consume any batch data
        # but for compatibility we keep the interface
        _ = batch.to_dict(orient="records")
        self.kg_instance.reload()
        self.chunk_storage.reload()

        yield from self.partition()

    def partition(self) -> Iterable[pd.DataFrame]:
        method = self.partition_kwargs["method"]
        method_params = self.partition_kwargs["method_params"]
        if method == "bfs":
            logger.info("Partitioning knowledge graph using BFS method.")
            partitioner = BFSPartitioner()
        elif method == "dfs":
            logger.info("Partitioning knowledge graph using DFS method.")
            partitioner = DFSPartitioner()
        elif method == "ece":
            logger.info("Partitioning knowledge graph using ECE method.")
            # TODO： before ECE partitioning, we need to:
            # 1. 'quiz' and 'judge' to get the comprehension loss if unit_sampling is not random
            # 2. pre-tokenize nodes and edges to get the token length
            self._pre_tokenize()
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

        communities = partitioner.partition(g=self.kg_instance, **method_params)

        for community in communities:
            batch = partitioner.community2batch(community, g=self.kg_instance)
            batch = self._attach_additional_data_to_node(batch)

            yield pd.DataFrame(
                {
                    "nodes": [batch[0]],
                    "edges": [batch[1]],
                }
            )

    def _pre_tokenize(self) -> None:
        """Pre-tokenize all nodes and edges to add token length information."""
        logger.info("Starting pre-tokenization of nodes and edges...")

        nodes = self.kg_instance.get_all_nodes()
        edges = self.kg_instance.get_all_edges()

        # Process nodes
        for node_id, node_data in nodes:
            if "length" not in node_data:
                try:
                    description = node_data.get("description", "")
                    tokens = self.tokenizer_instance.encode(description)
                    node_data["length"] = len(tokens)
                    self.kg_instance.update_node(node_id, node_data)
                except Exception as e:
                    logger.warning("Failed to tokenize node %s: %s", node_id, e)
                    node_data["length"] = 0

        # Process edges
        for u, v, edge_data in edges:
            if "length" not in edge_data:
                try:
                    description = edge_data.get("description", "")
                    tokens = self.tokenizer_instance.encode(description)
                    edge_data["length"] = len(tokens)
                    self.kg_instance.update_edge(u, v, edge_data)
                except Exception as e:
                    logger.warning("Failed to tokenize edge %s-%s: %s", u, v, e)
                    edge_data["length"] = 0

        # Persist changes
        self.kg_instance.index_done_callback()
        logger.info("Pre-tokenization completed.")

    def _attach_additional_data_to_node(self, batch: tuple) -> tuple:
        """
        Attach additional data from chunk_storage to nodes in the batch.
        :param batch: tuple of (nodes_data, edges_data)
        :return: updated batch with additional data attached to nodes
        """
        nodes_data, edges_data = batch

        for node_id, node_data in nodes_data:
            entity_type = (node_data.get("entity_type") or "").lower()
            
            if not entity_type:
                continue

            source_ids = [
                sid.strip()
                for sid in node_data.get("source_id", "").split("<SEP>")
                if sid.strip()
            ]
            
            if not source_ids:
                continue

            # Handle images
            if "image" in entity_type:
                image_chunks = [
                    data
                    for sid in source_ids
                    if "image" in sid.lower()
                    and (data := self.chunk_storage.get_by_id(sid))
                ]
                if image_chunks:
                    # The generator expects a dictionary with an 'img_path' key, not a list of captions.
                    # We'll use the first image chunk found for this node.
                    node_data["image_data"] = json.loads(image_chunks[0]["content"])
                    logger.debug("Attached image data to node %s", node_id)
            
            # Handle omics data (protein/dna/rna)
            molecule_type = None
            if entity_type in ("protein", "dna", "rna"):
                molecule_type = entity_type
            else:
                # Infer from source_id prefix
                for sid in source_ids:
                    sid_lower = sid.lower()
                    if sid_lower.startswith("protein-"):
                        molecule_type = "protein"
                        break
                    elif sid_lower.startswith("dna-"):
                        molecule_type = "dna"
                        break
                    elif sid_lower.startswith("rna-"):
                        molecule_type = "rna"
                        break
            
            if molecule_type:
                omics_chunks = [
                    data
                    for sid in source_ids
                    if (data := self.chunk_storage.get_by_id(sid))
                ]
                
                if not omics_chunks:
                    logger.warning("No chunks found for node %s (type: %s) with source_ids: %s", node_id, molecule_type, source_ids)
                    continue
                
                first_chunk = omics_chunks[0]
                def get_chunk_value(field: str):
                    # First check root level of chunk
                    if field in first_chunk:
                        return first_chunk[field]
                    # Then check metadata if it exists and is a dict
                    chunk_metadata = first_chunk.get("metadata")
                    if isinstance(chunk_metadata, dict) and field in chunk_metadata:
                        return chunk_metadata[field]
                    return None
                
                # Attach sequence if not already present
                if "sequence" not in node_data:
                    sequence = get_chunk_value("sequence")
                    if sequence:
                        node_data["sequence"] = sequence
                    else:
                        logger.warning("No sequence found in chunk for node %s. Chunk keys: %s", node_id, list(first_chunk.keys())[:15])
                
                # Attach molecule_type if not present
                if "molecule_type" not in node_data:
                    chunk_molecule_type = get_chunk_value("molecule_type")
                    if chunk_molecule_type:
                        node_data["molecule_type"] = chunk_molecule_type
                
                # Attach molecule-specific fields
                field_mapping = {
                    "protein": ["protein_name", "gene_names", "organism", "function", "id", "database", "entry_name", "uniprot_id"],
                    "dna": ["gene_name", "gene_description", "organism", "chromosome", "genomic_location", "function", "gene_type", "id", "database"],
                    "rna": ["rna_type", "description", "organism", "related_genes", "gene_name", "so_term", "id", "database", "rnacentral_id"],
                }
                
                for field in field_mapping.get(molecule_type, []):
                    if field not in node_data:
                        value = get_chunk_value(field)
                        if value:
                            node_data[field] = value

        return nodes_data, edges_data
