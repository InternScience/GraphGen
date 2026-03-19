from collections import defaultdict
from typing import List

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import LightRAGKGBuilder
from graphgen.utils import run_concurrent


def build_text_kg(
    llm_client: BaseLLMWrapper,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    max_loop: int = 3,
    relation_confidence_threshold: float = 0.5,
    require_entity_evidence: bool = False,
    require_relation_evidence: bool = True,
    validate_evidence_in_source: bool = False,
) -> tuple:
    """
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param chunks
    :param max_loop: Maximum number of loops for entity and relationship extraction
    :param relation_confidence_threshold: Minimum confidence score for accepting a relation
    :param require_entity_evidence: If True, entities without evidence span are dropped
    :param require_relation_evidence: If True, relations without evidence span are dropped
    :param validate_evidence_in_source: If True, evidence spans must be found in the source chunk
    :return:
    """

    kg_builder = LightRAGKGBuilder(
        llm_client=llm_client,
        max_loop=max_loop,
        relation_confidence_threshold=relation_confidence_threshold,
        require_entity_evidence=require_entity_evidence,
        require_relation_evidence=require_relation_evidence,
        validate_evidence_in_source=validate_evidence_in_source,
    )

    results = run_concurrent(
        kg_builder.extract,
        chunks,
        desc="[2/4]Extracting entities and relationships from chunks",
        unit="chunk",
    )
    results = [res for res in results if res]

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v)

    nodes = run_concurrent(
        lambda kv: kg_builder.merge_nodes(kv, kg_instance=kg_instance),
        list(nodes.items()),
        desc="Inserting entities into storage",
    )

    edges = run_concurrent(
        lambda kv: kg_builder.merge_edges(kv, kg_instance=kg_instance),
        list(edges.items()),
        desc="Inserting relationships into storage",
    )

    return nodes, edges
