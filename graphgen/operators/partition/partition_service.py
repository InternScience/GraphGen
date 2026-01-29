import os
from typing import Iterable, Tuple

from graphgen.bases import BaseGraphStorage, BaseOperator, BaseTokenizer
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
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **partition_kwargs,
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="partition"
        )
        self.kg_instance: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        tokenizer_model = os.getenv("TOKENIZER_MODEL", "cl100k_base")
        self.tokenizer_instance: BaseTokenizer = Tokenizer(model_name=tokenizer_model)
        method = partition_kwargs["method"]
        self.method_params = partition_kwargs["method_params"]

        if method == "bfs":
            self.partitioner = BFSPartitioner()
        elif method == "dfs":
            self.partitioner = DFSPartitioner()
        elif method == "ece":
            # before ECE partitioning, we need to:
            # 'quiz' and 'judge' to get the comprehension loss if unit_sampling is not random
            self.partitioner = ECEPartitioner()
        elif method == "leiden":
            self.partitioner = LeidenPartitioner()
        elif method == "anchor_bfs":
            self.partitioner = AnchorBFSPartitioner(
                anchor_type=self.method_params.get("anchor_type"),
                anchor_ids=set(self.method_params.get("anchor_ids", []))
                if self.method_params.get("anchor_ids")
                else None,
            )
        else:
            raise ValueError(f"Unsupported partition method: {method}")

    def process(self, batch: list) -> Tuple[Iterable[list], dict]:
        # this operator does not consume any batch data
        # but for compatibility we keep the interface
        self.kg_instance.reload()

        communities: Iterable = self.partitioner.partition(
            g=self.kg_instance, **self.method_params
        )

        def generator():
            count = 0
            for community in communities:
                count += 1
                batch = self.partitioner.community2batch(community, g=self.kg_instance)
                # batch = self._attach_additional_data_to_node(batch)

                result = {
                    "nodes": batch[0],
                    "edges": batch[1],
                }
                result["_trace_id"] = self.get_trace_id(result)
                yield result
            logger.info("Total communities partitioned: %d", count)

        return generator(), {}

    # def _attach_additional_data_to_node(self, batch: tuple) -> tuple:
    #     """
    #     Attach additional data from chunk_storage to nodes in the batch.
    #     :param batch: tuple of (nodes_data, edges_data)
    #     :return: updated batch with additional data attached to nodes
    #     """
    #     nodes_data, edges_data = batch
    #
    #     for node_id, node_data in nodes_data:
    #         entity_type = (node_data.get("entity_type") or "").lower()
    #         if not entity_type:
    #             continue
    #
    #         source_ids = [
    #             sid.strip()
    #             for sid in node_data.get("source_id", "").split("<SEP>")
    #             if sid.strip()
    #         ]
    #
    #         # Handle images
    #         if "image" in entity_type:
    #             image_chunks = [
    #                 data
    #                 for sid in source_ids
    #                 if "image" in sid.lower()
    #                 and (data := self.chunk_storage.get_by_id(sid))
    #             ]
    #             if image_chunks:
    #                 # The generator expects a dictionary with an 'img_path' key, not a list of captions.
    #                 # We'll use the first image chunk found for this node.
    #                 node_data["image_data"] = json.loads(image_chunks[0]["content"])
    #                 logger.debug("Attached image data to node %s", node_id)
    #
    #     return nodes_data, edges_data
