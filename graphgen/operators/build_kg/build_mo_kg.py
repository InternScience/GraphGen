import json
from typing import List

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.templates import PROTEIN_ANCHOR_PROMPT, PROTEIN_KG_EXTRACTION_PROMPT
from graphgen.utils import detect_main_language, logger, run_concurrent


async def build_mo_kg(
    llm_client: BaseLLMWrapper,
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

    async def extract_mo_info(chunk: Chunk):
        content = chunk.content
        language = detect_main_language(content)
        prompt = PROTEIN_ANCHOR_PROMPT[language].format(chunk=content)
        result = await llm_client.generate_answer(prompt)
        try:
            json_result = json.loads(result)
            return json_result
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response: %s", result)
            return {}

    results = await run_concurrent(
        extract_mo_info,
        chunks,
        desc="Extracting multi-omics anchoring information from chunks",
        unit="chunk",
        progress_bar=progress_bar,
    )
    # Merge results
    from collections import defaultdict

    bags = defaultdict(set)
    for item in results:
        for k, v in item.items():
            if v is not None and str(v).strip():
                bags[k].add(str(v).strip())

    merged = {
        k: " | ".join(sorted(v)) if len(v) > 1 else next(iter(v))
        for k, v in bags.items()
    }

    # TODO: search database for more info
    # try:
    #     search_results = await search(merged["Protein accession or ID"])
    # except Exception as e:
    #     logger.warning("Failed to search for protein info: %s", e)
    #     search_results = {}

    mo_text = "\n".join([f"{k}: {v}" for k, v in merged.items()])
    lang = detect_main_language(mo_text)
    prompt = PROTEIN_KG_EXTRACTION_PROMPT[lang].format(
        input_text=mo_text,
        **PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"],
    )
    kg_output = await llm_client.generate_answer(prompt)
    print(kg_output)
    # TODO: parse kg_output and insert into kg_instance
    return kg_instance
