import json
import re
from typing import List

import gradio as gr

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import OpenAIClient
from graphgen.templates import PROTEIN_ANCHOR_PROMPT, PROTEIN_KG_EXTRACTION_PROMPT
from graphgen.utils import (
    detect_main_language,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    logger,
    run_concurrent,
    split_string_by_multi_markers,
)


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

    # 组织成文本
    mo_text = "\n".join([f"{k}: {v}" for k, v in merged.items()])
    lang = detect_main_language(mo_text)
    prompt = PROTEIN_KG_EXTRACTION_PROMPT[lang].format(
        input_text=mo_text,
        **PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"],
    )
    kg_output = await llm_client.generate_answer(prompt)

    logger.debug("Image chunk extraction result: %s", kg_output)

    # parse the result
    records = split_string_by_multi_markers(
        kg_output,
        [
            PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"]["record_delimiter"],
            PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"]["completion_delimiter"],
        ],
    )

    print(records)
    raise NotImplementedError

    nodes = defaultdict(list)
    edges = defaultdict(list)

    for record in records:
        match = re.search(r"\((.*)\)", record)
        if not match:
            continue
        inner = match.group(1)

        attributes = split_string_by_multi_markers(
            inner, [PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"]["tuple_delimiter"]]
        )

        entity = await handle_single_entity_extraction(attributes, "temp")
        if entity is not None:
            nodes[entity["entity_name"]].append(entity)
            continue

        relation = await handle_single_relationship_extraction(attributes, "temp")
        if relation is not None:
            key = (relation["src_id"], relation["tgt_id"])
            edges[key].append(relation)
