from typing import List

import gradio as gr

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import OpenAIClient


async def build_mo_kg(
    llm_client: OpenAIClient,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    progress_bar: gr.Progress = None,
):
    """
    Build multi-omics KG and merge into kg_instance. (Multi-Omics: genomics, proteomics, metabolomics, etc.)
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param chunks
    :param progress_bar: Gradio progress bar to show the progress of the extraction
    :return:
    """
    # TODO: implement multi-omics KG building logic here
    print("llm_client:", llm_client)
    print("kg_instance:", kg_instance)
    print("chunks:", chunks)
    print("progress_bar:", progress_bar)
    return kg_instance
