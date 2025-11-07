import json

import gradio as gr

from graphgen.bases import BaseKVStorage, BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.models.extractor import SchemaGuidedExtractor
from graphgen.utils import logger, run_concurrent


async def extract_info(
    llm_client: BaseLLMWrapper,
    chunk_storage: BaseKVStorage,
    extract_config: dict,
    progress_bar: gr.Progress = None,
):
    """
    Extract information from chunks
    :param llm_client: LLM client
    :param chunk_storage: storage for chunks
    :param extract_config
    :param progress_bar
    :return: extracted information
    """

    method = extract_config.get("method")
    if method == "schema_guided":
        schema_file = extract_config.get("schema_file")
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        extractor = SchemaGuidedExtractor(llm_client, schema)
    else:
        raise ValueError(f"Unsupported extraction method: {method}")

    chunks = await chunk_storage.get_all()
    chunks = [{k: v} for k, v in chunks.items()]
    logger.info(f"Start extracting information from {len(chunks)} chunks")

    results = await run_concurrent(
        extractor.extract,
        chunks,
        desc="Extracting information",
        unit="chunk",
        progress_bar=progress_bar,
    )
    print(results)

    # TODO: 对results合并，去重

    return []
