from typing import Dict, List, Tuple

from graphgen.bases import Chunk

from .light_rag_kg_builder import LightRAGKGBuilder


class MOKGBuilder(LightRAGKGBuilder):
    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """
        Multi-Omics Knowledge Graph Builder
        Step1: Extract and output a JSON object containing protein information from the given chunk.
        Step2: Get more details about the protein by querying external databases if necessary.
        Step3: Construct entities and relationships for the protein knowledge graph.
        Step4: Return the entities and relationships.
        :param chunk
        :return: Tuple containing entities and relationships.
        """
        # TODO: Implement the multi-omics KG extraction logic here
        print(chunk)
        return {}, {}
