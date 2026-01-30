from typing import Tuple

from graphgen.bases import BaseOperator
from graphgen.utils import logger


class FilterService(BaseOperator):
    def __init__(
        self, working_dir: str = "cache", kv_backend: str = "rocksdb", **filter_kwargs
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="filter"
        )
        method = filter_kwargs["method"]
        method_params = filter_kwargs["method_params"]
        self.metric = method_params["metric"]
        if method == "range":
            from graphgen.models import RangeFilter

            self.filter_instance = RangeFilter(
                min_val=method_params["min_val"],
                max_val=method_params["max_val"],
                left_inclusive=method_params.get("left_inclusive", True),
                right_inclusive=method_params.get("right_inclusive", True),
            )
        else:
            raise ValueError(f"Unsupported filter method: {method}")

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Filter the items in the batch.
        :return: A tuple of (results, meta_updates)
            results: A list of filtered items.
            meta_updates: empty as filtering does not create new items.
        """
        results = []
        meta_updates = {}

        for item in batch:
            value = item["metrics"].get(self.metric)
            if value is None:
                logger.warning(
                    f"Item {item} does not have metric {self.metric}. Skipping."
                )
                continue
            if self.filter_instance.filter(value):
                results.append(item)

        return results, meta_updates
