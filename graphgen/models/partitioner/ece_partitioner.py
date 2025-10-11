from typing import Any, List

from graphgen.bases import BaseGraphStorage, BasePartitioner
from graphgen.bases.datatypes import Community


class ECEPartitioner(BasePartitioner):
    def partition(
        self,
        g: BaseGraphStorage,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> List[Community]:
        pass

    def split_communities(self, communities: List[Community]) -> List[Community]:
        pass
