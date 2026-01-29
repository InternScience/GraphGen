import inspect
import os
from abc import ABC, abstractmethod
from typing import Iterable, Union, Tuple

import pandas as pd
import ray


class BaseOperator(ABC):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        op_name: str = None,
    ):
        # lazy import to avoid circular import
        from graphgen.common import init_storage
        from graphgen.utils import set_logger

        log_dir = os.path.join(working_dir, "logs")
        self.op_name = op_name or self.__class__.__name__
        self.working_dir = working_dir
        self.kv_storage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace=self.op_name
        )

        try:
            ctx = ray.get_runtime_context()
            worker_id = ctx.get_actor_id() or ctx.get_worker_id()
            worker_id_short = worker_id[-6:] if worker_id else "driver"
        except Exception as e:
            print(
                "Warning: Could not get Ray worker ID, defaulting to 'local'. Exception:",
                e,
            )
            worker_id_short = "local"

        # e.g. cache/logs/ChunkService_a1b2c3.log
        log_file = os.path.join(log_dir, f"{self.op_name}_{worker_id_short}.log")

        self.logger = set_logger(
            log_file=log_file, name=f"{self.op_name}.{worker_id_short}", force=True
        )

        self.logger.info(
            "[%s] Operator initialized on Worker %s", self.op_name, worker_id_short
        )

    def __call__(
        self, batch: pd.DataFrame
    ) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
        # lazy import to avoid circular import
        from graphgen.utils import CURRENT_LOGGER_VAR

        logger_token = CURRENT_LOGGER_VAR.set(self.logger)
        try:
            self.kv_storage.reload()
            to_process, recovered = self.split(batch)
            # yield recovered chunks first
            if not recovered.empty:
                yield recovered

            if to_process.empty:
                return

            data = to_process.to_dict(orient="records")
            result, meta_update = self.process(data)
            if inspect.isgenerator(result):
                is_first = True
                for res in result:
                    yield pd.DataFrame([res])
                    self.store([res], meta_update if is_first else {})
                    is_first = False
            else:
                yield pd.DataFrame(result)
                self.store(result, meta_update)
        finally:
            CURRENT_LOGGER_VAR.reset(logger_token)

    def get_logger(self):
        return self.logger

    def get_meta_forward(self):
        return self.kv_storage.get_by_id("_meta_forward") or {}

    def get_meta_inverse(self):
        return self.kv_storage.get_by_id("_meta_inverse") or {}

    def get_trace_id(self, content: dict) -> str:
        from graphgen.utils import compute_dict_hash

        return compute_dict_hash(content, prefix=f"{self.op_name}-")

    def split(self, batch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the input batch into to_process & processed based on _meta data in KV_storage
        :param batch
        :return:
            to_process: DataFrame of documents to be chunked
            recovered: Result DataFrame of already chunked documents
        """
        meta_forward = self.get_meta_forward()
        meta_ids = set(meta_forward.keys())
        mask = batch["_trace_id"].isin(meta_ids)
        to_process = batch[~mask]
        processed = batch[mask]

        if processed.empty:
            return to_process, pd.DataFrame()

        all_ids = [
            pid for tid in processed["_trace_id"] for pid in meta_forward.get(tid, [])
        ]

        recovered_chunks = self.kv_storage.get_by_ids(all_ids)
        recovered_chunks = [c for c in recovered_chunks if c is not None]
        return to_process, pd.DataFrame(recovered_chunks)

    def store(self, results: list, meta_update: dict):
        batch = {res["_trace_id"]: res for res in results}
        self.kv_storage.upsert(batch)

        # update forward meta
        forward_meta = self.get_meta_forward()
        forward_meta.update(meta_update)
        self.kv_storage.update({"_meta_forward": forward_meta})

        # update inverse meta
        inverse_meta = self.get_meta_inverse()
        for k, v_list in meta_update.items():
            for v in v_list:
                inverse_meta[v] = k
        self.kv_storage.update({"_meta_inverse": inverse_meta})
        self.kv_storage.index_done_callback()

    @abstractmethod
    def process(self, batch: list) -> Tuple[Union[list, Iterable[list]], dict]:
        """
        Process the input batch and return the result.
        :param batch
        :return:
            result: DataFrame of processed documents
            meta_update: dict of meta data to be updated
        """
