from dataclasses import dataclass
from typing import Dict, List, Tuple

from graphgen.bases import BaseGraphStorage, BaseKGBuilder, BaseLLMClient, Chunk


@dataclass
class MMKGBuilder(BaseKGBuilder):
    llm_client: BaseLLMClient = None

    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        pass

    async def merge_nodes(
        self, node_data: tuple[str, List[dict]], kg_instance: BaseGraphStorage
    ) -> None:
        pass

    async def merge_edges(
        self,
        edges_data: tuple[Tuple[str, str], List[dict]],
        kg_instance: BaseGraphStorage,
    ) -> None:
        pass
