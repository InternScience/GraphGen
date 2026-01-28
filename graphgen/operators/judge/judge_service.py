import math

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common import init_llm, init_storage
from graphgen.templates import STATEMENT_JUDGEMENT_PROMPT
from graphgen.utils import logger, run_concurrent, yes_no_loss_entropy


class JudgeService(BaseOperator):
    """Service for judging graph edges and nodes using a trainee LLM."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="judge"
        )
        self.llm_client: BaseLLMWrapper = init_llm("trainee")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )

    async def _process_single_judge(self, item: dict) -> dict:
        description = item["description"]
        try:
            judgement = await self.llm_client.generate_topk_per_token(
                STATEMENT_JUDGEMENT_PROMPT["TEMPLATE"].format(statement=description)
            )
            top_candidates = judgement[0].top_candidates
            gt = item.get("ground_truth", "yes")
            loss = yes_no_loss_entropy([top_candidates], [gt])
            logger.debug("Description: %s Loss: %s", description, loss)
            item["loss"] = loss
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in judging description: %s", e)
            logger.info("Use default loss 0.1")
            item["loss"] = -math.log(0.1)
        return item

    def process(self, batch: list) -> pd.DataFrame:
        """
        Judge the description in the item and compute the loss.
        """
        self.graph_storage.reload()

        results = run_concurrent(
            self._process_single_judge,
            batch,
            desc="Judging descriptions",
            unit="description",
        )

        to_store = []
        meta_update = {}

        for input_trace_id, result in zip(
            [item["_trace_id"] for item in batch], results
        ):
            if not result:
                continue
            index = result["index"]
            loss = result["loss"]
            if isinstance(index, str):
                node_id = index
                node_data = self.graph_storage.get_node(node_id)
                node_data["loss"] = loss
                self.graph_storage.update_node(node_id, node_data)
            elif isinstance(index, tuple):
                edge_source, edge_target = index
                edge_data = self.graph_storage.get_edge(edge_source, edge_target)
                edge_data["loss"] = loss
                self.graph_storage.update_edge(edge_source, edge_target, edge_data)

            result["_trace_id"] = self.generate_trace_id(result)
            to_store.append(result)
            meta_update.setdefault(input_trace_id, []).append(result["_trace_id"])
        self.graph_storage.index_done_callback()
        self.store(to_store, meta_update)

        return pd.DataFrame(results)
