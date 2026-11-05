import json
import math
from typing import Any, Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
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

    def _normalize_unit_index(self, value: Any) -> Any:
        """传输层还原：JSON 字符串 / list → tuple（图存储写回分派依据）。"""
        if isinstance(value, str) and value.startswith("["):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list) and len(decoded) == 2:
                return tuple(decoded)
        if isinstance(value, list) and len(value) == 2:
            return tuple(value)
        return value

    def _write_back_losses(self, records: list) -> int:
        """缓存和新判共用一次执行方原子写回；失败必须传播，不能跳过落图。"""
        updates = []
        for result in records:
            if not isinstance(result, dict) or not result:
                continue
            index = self._normalize_unit_index(result.get("index"))
            loss = result.get("loss")
            if index is None or not isinstance(loss, (int, float)):
                continue
            if isinstance(index, str) or (isinstance(index, tuple) and len(index) == 2):
                updates.append((index, loss))
        return self.graph_storage.write_losses(updates)

    def split(self, batch):
        # 缓存命中的记录由基类直接 yield，绕过 process()——这里补上它们的
        # 图写回（执行方单 RPC 事务，允许多 judge 副本）。
        to_process, recovered = super().split(batch)
        if not recovered.empty:
            written = self._write_back_losses(recovered.to_dict(orient="records"))
            self.logger.info(
                "[judge] wrote %d cached losses back to graph storage", written
            )
            # PMS_PATCHES.md #16：缓存恢复行同样遵守对外 index=JSON 字符串口径。
            # 历史 KV 缓存（1.2.0 之前的 quiz→judge 链路）把边 index 存成 list，
            # 原样 yield 会与新产出混出 str/list 混型列，Arrow 推断直接失败。
            for row_position, row in recovered.iterrows():
                value = row.get("index")
                if isinstance(value, (list, tuple)):
                    recovered.at[row_position, "index"] = json.dumps(
                        list(value), ensure_ascii=False
                    )
        return to_process, recovered

    async def _process_single_judge(self, item: dict) -> dict:
        # PMS_PATCHES.md #13/#16：跨 operator 的 Arrow/pandas 边界会把二元组 index
        # 变成 list，graph_units 传输层进一步把边 index 编码为 JSON 字符串
        # （混合 str/list 列会炸 Arrow tensor 分支）。这里统一还原成 tuple，
        # 供下方 storage 写回按 isinstance(index, tuple) 分派。
        # 归一逻辑只有 _normalize_unit_index 一份实现，两条入口共用。
        normalized = self._normalize_unit_index(item.get("index"))
        if normalized is not item.get("index"):
            item["index"] = normalized
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

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Judge the description in the item and compute the loss.
        """
        results = run_concurrent(
            self._process_single_judge,
            batch,
            desc="Judging descriptions",
            unit="description",
        )

        # 不在 judge 进程拆分 reload/update/flush RPC；一次 actor 调用保证
        # partition、graph_units 的 reload 不能插入未落盘更新之间。
        self._write_back_losses(results)
        to_store = []
        meta_update = {}

        for input_trace_id, result in zip(
            [item["_trace_id"] for item in batch], results
        ):
            if not result:
                continue
            result["_trace_id"] = self.get_trace_id(result)
            to_store.append(result)
            meta_update.setdefault(input_trace_id, []).append(result["_trace_id"])

        # PMS_PATCHES.md #16：图写回已完成，对外产出把边 index 编码回 JSON
        # 字符串（与 graph_units 传输口径一致；列保持纯 str，Arrow 安全）。
        # run_concurrent 对异常项留 None 占位（如输入行缺 description），
        # 与上方写回循环同款守卫，不得让占位崩掉整批产出。
        for result in results:
            if not isinstance(result, dict) or not result:
                continue
            if isinstance(result.get("index"), (tuple, list)):
                result["index"] = json.dumps(list(result["index"]), ensure_ascii=False)

        return results, meta_update
