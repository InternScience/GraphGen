from typing import List

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.models.extractor import SchemaGuidedExtractor
from graphgen.utils import logger, run_concurrent


async def extract(
    llm_client: BaseLLMWrapper,
    chunks: List[Chunk],
    generation_config: dict,
    progress_bar: gr.Progress = None,
):
    """
    Extract information from chunks
    :param llm_client: LLM client
    :param chunks
    :param generation_config
    :param progress_bar
    :return: extracted information
    """

    method = generation_config.get("method")
    if method == "schema_guided":
        schema = generation_config.get("schema")
        extractor = SchemaGuidedExtractor(llm_client, schema)
        print(extractor)
    else:
        raise ValueError(f"Unsupported extraction method: {method}")

    logger.info("[Extraction] method: %s, chunks: %d", method, len(chunks))

    # results = await run_concurrent(
    #     extractor.extract,
    #     [chunk.content for chunk in chunks],
    #     desc="Extracting information",
    #     unit="chunk",
    #     progress_bar=progress_bar,
    # )
    #
    # # TODO: 对results合并，去重
    # return results

    return []
