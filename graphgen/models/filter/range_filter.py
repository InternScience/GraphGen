from typing import Union

import numpy as np

from graphgen.bases import BaseValueFilter


class RangeFilter(BaseValueFilter):
    """
    keeps values within a specified range [min_val, max_val] (inclusive or exclusive)
    """

    def __init__(
        self,
        min_val: float,
        max_val: float,
        left_inclusive: bool = True,
        right_inclusive: bool = True,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.left_inclusive = left_inclusive
        self.right_inclusive = right_inclusive

    def filter(self, data: Union[int, float, np.number]) -> bool:
        value = float(data)
        if self.left_inclusive and self.right_inclusive:
            return self.min_val <= value <= self.max_val
        if self.left_inclusive and not self.right_inclusive:
            return self.min_val <= value < self.max_val
        if not self.left_inclusive and self.right_inclusive:
            return self.min_val < value <= self.max_val
        return self.min_val < value < self.max_val

    @property
    def filter_type(self) -> str:
        return "range"

    def __repr__(self) -> str:
        return f"RangeFilter({self.min_val}, {self.max_val})"
