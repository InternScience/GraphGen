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
        self.method_params = partition_kwargs.get("method_params", {})
        # PMS_PATCHES.md #15：max_communities 截断分区输出——loss 采样下分区按
        # 种子单元 loss 降序产出社区，取前 N 即"盲区选题"；random 采样下为任意
        # 抽样。None（官方默认）= 全量社区，行为不变。
        self.max_communities: int | None = partition_kwargs.get("max_communities")
        if method == "event_join":
            self.method_params = dict(self.method_params)
            self.method_params.setdefault("event_entity_type", partition_kwargs.get("event_entity_type", "EVENT"))
            self.method_params.setdefault("anchor_attribute", partition_kwargs.get("anchor_attribute", "anchor"))
            self.method_params.setdefault("require_same_anchor", partition_kwargs.get("require_same_anchor", True))

        if method == "bfs":
            from graphgen.models import BFSPartitioner

            self.partitioner = BFSPartitioner()
        elif method == "dfs":
            from graphgen.models import DFSPartitioner

            self.partitioner = DFSPartitioner()
        elif method == "event_join":
            from graphgen.models import EventJoinPartitioner

            self.partitioner = EventJoinPartitioner()
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
        elif method == "triple":
            from graphgen.models import TriplePartitioner

            self.partitioner = TriplePartitioner()
        elif method == "quintuple":
            from graphgen.models import QuintuplePartitioner

            self.partitioner = QuintuplePartitioner()
        else:
            raise ValueError(f"Unsupported partition method: {method}")

    def process(self, batch: list) -> Tuple[Iterable[dict], dict]:
        # this operator does not consume any batch data
        # but for compatibility we keep the interface
        # PMS_PATCHES.md #16：上游 index_done_callback 整文件写 graphml 与本处
        # reload 之间存在窗口（ParseError 无明细）；有界重试读完整文件。
        import time

        for attempt in range(3):
            try:
                self.kg_instance.reload()
                break
            except Exception as exc:  # pylint: disable=broad-except
                if attempt == 2:
                    raise
                logger.warning(
                    "[partition] kg reload failed (attempt %d/3): %s: %s",
                    attempt + 1, type(exc).__name__, str(exc)[:200],
                )
                time.sleep(10)

        communities: Iterable = self.partitioner.partition(
            g=self.kg_instance, **self.method_params
        )

        def generator():
            count = 0
            for community in communities:
                if self.max_communities is not None and count >= self.max_communities:
                    break
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
