from collections import defaultdict
from typing import List

import gradio as gr

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import LightRAGKGBuilder, OpenAIClient, Tokenizer
from graphgen.operators.build_kg.merge_kg import merge_edges, merge_nodes
from graphgen.utils import run_concurrent


async def build_kg(
    llm_client: OpenAIClient,
    kg_instance: BaseGraphStorage,
    tokenizer_instance: Tokenizer,
    chunks: List[Chunk],
    progress_bar: gr.Progress = None,
):
    """
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param tokenizer_instance
    :param chunks
    :param progress_bar: Gradio progress bar to show the progress of the extraction
    :return:
    """

    kg_builder = LightRAGKGBuilder(llm_client=llm_client, max_loop=3)

    results = await run_concurrent(
        kg_builder.extract,
        chunks,
        desc="[2/4]Extracting entities and relationships from chunks",
        unit="chunk",
        progress_bar=progress_bar,
    )

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v)

    await merge_nodes(nodes, kg_instance, llm_client, tokenizer_instance)
    await merge_edges(edges, kg_instance, llm_client, tokenizer_instance)

    return kg_instance
