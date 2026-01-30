import os
from typing import Iterable, Tuple

from graphgen.bases import BaseGraphStorage, BaseOperator, BaseTokenizer
from graphgen.common.init_storage import init_storage
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

        from graphgen.models import Tokenizer

        self.tokenizer_instance: BaseTokenizer = Tokenizer(model_name=tokenizer_model)
        method = partition_kwargs["method"]
        self.method_params = partition_kwargs["method_params"]

        if method == "bfs":
            from graphgen.models import BFSPartitioner

            self.partitioner = BFSPartitioner()
        elif method == "dfs":
            from graphgen.models import DFSPartitioner

            self.partitioner = DFSPartitioner()
        elif method == "ece":
            # before ECE partitioning, we need to:
            # 'quiz' and 'judge' to get the comprehension loss if unit_sampling is not random
            from graphgen.models import ECEPartitioner

            self.partitioner = ECEPartitioner()
        elif method == "leiden":
            from graphgen.models import LeidenPartitioner

            self.partitioner = LeidenPartitioner()
        elif method == "anchor_bfs":
            from graphgen.models import AnchorBFSPartitioner

            self.partitioner = AnchorBFSPartitioner(
                anchor_type=self.method_params.get("anchor_type"),
                anchor_ids=set(self.method_params.get("anchor_ids", []))
                if self.method_params.get("anchor_ids")
                else None,
            )
        else:
            raise ValueError(f"Unsupported partition method: {method}")

    def process(self, batch: list) -> Tuple[Iterable[dict], dict]:
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
                b = self.partitioner.community2batch(community, g=self.kg_instance)

                result = {
                    "nodes": b[0],
                    "edges": b[1],
                }
                result["_trace_id"] = self.get_trace_id(result)
                yield result
            logger.info("Total communities partitioned: %d", count)

        return generator(), {}
