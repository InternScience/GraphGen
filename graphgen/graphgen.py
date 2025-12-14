import hashlib
import os
import time
from typing import Dict

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.models import (
    JsonKVStorage,
    JsonListStorage,
    NetworkXStorage,
    OpenAIClient,
    Tokenizer,
)
from graphgen.operators import (
    build_kg,
    chunk_documents,
    extract_info,
    generate_qas,
    init_llm,
    judge_statement,
    partition_kg,
    quiz,
    read_files,
    search_all,
)
from graphgen.utils import async_to_sync_method, compute_mm_hash, logger

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class GraphGen:
    def __init__(
        self,
        unique_id: int = int(time.time()),
        working_dir: str = os.path.join(sys_path, "cache"),
        tokenizer_instance: Tokenizer = None,
        synthesizer_llm_client: OpenAIClient = None,
        trainee_llm_client: OpenAIClient = None,
        progress_bar: gr.Progress = None,
    ):
        self.unique_id: int = unique_id
        self.working_dir: str = working_dir

        # llm
        self.tokenizer_instance: Tokenizer = tokenizer_instance or Tokenizer(
            model_name=os.getenv("TOKENIZER_MODEL", "cl100k_base")
        )

        self.synthesizer_llm_client: BaseLLMWrapper = (
            synthesizer_llm_client or init_llm("synthesizer")
        )
        self.trainee_llm_client: BaseLLMWrapper = trainee_llm_client

        self.full_docs_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="full_docs"
        )
        self.chunks_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="chunks"
        )
        self.graph_storage: NetworkXStorage = NetworkXStorage(
            self.working_dir, namespace="graph"
        )
        self.rephrase_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="rephrase"
        )
        self.partition_storage: JsonListStorage = JsonListStorage(
            self.working_dir, namespace="partition"
        )
        self.search_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", "graphgen", f"{self.unique_id}"),
            namespace="search",
        )
        self.qa_storage: JsonListStorage = JsonListStorage(
            os.path.join(self.working_dir, "data", "graphgen", f"{self.unique_id}"),
            namespace="qa",
        )
        self.extract_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", "graphgen", f"{self.unique_id}"),
            namespace="extraction",
        )

        # webui
        self.progress_bar: gr.Progress = progress_bar

    @async_to_sync_method
    async def read(self, read_config: Dict):
        """
        read files from input sources with batch processing
        """
        # Get batch_size from config, default to 10000
        batch_size = read_config.pop("batch_size", 10000)
        
        doc_stream = read_files(**read_config, cache_dir=self.working_dir)

        batch = {}
        total_processed = 0
        
        for doc in doc_stream:
            doc_id = compute_mm_hash(doc, prefix="doc-")
            batch[doc_id] = doc
            
            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                _add_doc_keys = self.full_docs_storage.filter_keys(list(batch.keys()))
                new_docs = {k: v for k, v in batch.items() if k in _add_doc_keys}
                if new_docs:
                    self.full_docs_storage.upsert(new_docs)
                    total_processed += len(new_docs)
                    logger.info("Processed batch: %d new documents (total: %d)", len(new_docs), total_processed)
                batch.clear()

        # TODO: configurable whether to use coreference resolution

        # Process remaining documents in batch
        if batch:
            _add_doc_keys = self.full_docs_storage.filter_keys(list(batch.keys()))
            new_docs = {k: v for k, v in batch.items() if k in _add_doc_keys}
            if new_docs:
                self.full_docs_storage.upsert(new_docs)
                total_processed += len(new_docs)
                logger.info("Processed final batch: %d new documents (total: %d)", len(new_docs), total_processed)
        
        if total_processed == 0:
            logger.warning("All documents are already in the storage")
        else:
            self.full_docs_storage.index_done_callback()

    @async_to_sync_method
    async def chunk(self, chunk_config: Dict):
        """
        chunk documents into smaller pieces from full_docs_storage if not already present
        """

        new_docs = self.full_docs_storage.get_all()
        if len(new_docs) == 0:
            logger.warning("All documents are already in the storage")
            return

        inserting_chunks = await chunk_documents(
            new_docs,
            self.tokenizer_instance,
            self.progress_bar,
            **chunk_config,
        )

        _add_chunk_keys = self.chunks_storage.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return

        self.chunks_storage.upsert(inserting_chunks)
        self.chunks_storage.index_done_callback()

    @async_to_sync_method
    async def build_kg(self):
        """
        build knowledge graph from text chunks
        """
        # Step 1: get new chunks
        inserting_chunks = self.chunks_storage.get_all()

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return

        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))
        # Step 2: build knowledge graph from new chunks
        _add_entities_and_relations = await build_kg(
            llm_client=self.synthesizer_llm_client,
            kg_instance=self.graph_storage,
            chunks=[Chunk.from_dict(k, v) for k, v in inserting_chunks.items()],
            progress_bar=self.progress_bar,
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted from text chunks")
            return

        # Step 3: upsert new entities and relations to the graph storage
        self.graph_storage.index_done_callback()

        return _add_entities_and_relations

    @async_to_sync_method
    async def search(self, search_config: Dict):
        logger.info("[Search] %s ...", ", ".join(search_config["data_sources"]))

        # Get search_batch_size from config (default: 10000)
        search_batch_size = search_config.get("search_batch_size", 10000)
        
        # Get save_interval from config (default: 1000, 0 to disable)
        save_interval = search_config.get("save_interval", 1000)
        
        # Process in batches to avoid OOM
        all_flattened_results = {}
        batch_num = 0
        
        for seeds_batch in self.full_docs_storage.iter_batches(batch_size=search_batch_size):
            if len(seeds_batch) == 0:
                continue
                
            batch_num += 1
            logger.info("Processing search batch %d with %d documents", batch_num, len(seeds_batch))
            
            search_results = await search_all(
                seed_data=seeds_batch,
                search_config=search_config,
                search_storage=self.search_storage if save_interval > 0 else None,
                save_interval=save_interval,
            )

            # Convert search_results from {data_source: [results]} to {key: result}
            # This maintains backward compatibility
            for data_source, result_list in search_results.items():
                if not isinstance(result_list, list):
                    continue
                for result in result_list:
                    if result is None:
                        continue
                    # Use _search_query as key if available, otherwise generate a key
                    if isinstance(result, dict) and "_search_query" in result:
                        query = result["_search_query"]
                        key = f"{data_source}:{query}"
                    else:
                        # Generate a unique key
                        result_str = str(result)
                        key_hash = hashlib.md5(result_str.encode()).hexdigest()[:8]
                        key = f"{data_source}:{key_hash}"
                    all_flattened_results[key] = result

        if len(all_flattened_results) == 0:
            logger.warning("No search results generated")
            return

        _add_search_keys = self.search_storage.filter_keys(list(all_flattened_results.keys()))
        search_results = {
            k: v for k, v in all_flattened_results.items() if k in _add_search_keys
        }
        if len(search_results) == 0:
            logger.warning("All search results are already in the storage")
            return
        
        # Only save if not using periodic saving (to avoid duplicate saves)
        if save_interval == 0:
            self.search_storage.upsert(search_results)
            self.search_storage.index_done_callback()
        else:
            # Results were already saved periodically, just update index
            self.search_storage.index_done_callback()

    @async_to_sync_method
    async def quiz_and_judge(self, quiz_and_judge_config: Dict):
        logger.warning(
            "Quiz and Judge operation needs trainee LLM client."
            " Make sure to provide one."
        )
        max_samples = quiz_and_judge_config["quiz_samples"]
        await quiz(
            self.synthesizer_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            max_samples,
            progress_bar=self.progress_bar,
        )

        # TODO： assert trainee_llm_client is valid before judge
        if not self.trainee_llm_client:
            # TODO: shutdown existing synthesizer_llm_client properly
            logger.info("No trainee LLM client provided, initializing a new one.")
            self.synthesizer_llm_client.shutdown()
            self.trainee_llm_client = init_llm("trainee")

        re_judge = quiz_and_judge_config["re_judge"]
        _update_relations = await judge_statement(
            self.trainee_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            re_judge,
            progress_bar=self.progress_bar,
        )

        self.rephrase_storage.index_done_callback()
        _update_relations.index_done_callback()

        logger.info("Shutting down trainee LLM client.")
        self.trainee_llm_client.shutdown()
        self.trainee_llm_client = None
        logger.info("Restarting synthesizer LLM client.")
        self.synthesizer_llm_client.restart()

    @async_to_sync_method
    async def partition(self, partition_config: Dict):
        batches = await partition_kg(
            self.graph_storage,
            self.chunks_storage,
            self.tokenizer_instance,
            partition_config,
        )
        self.partition_storage.upsert(batches)
        return batches

    @async_to_sync_method
    async def extract(self, extract_config: Dict):
        logger.info("Extracting information from given chunks...")

        results = await extract_info(
            self.synthesizer_llm_client,
            self.chunks_storage,
            extract_config,
            progress_bar=self.progress_bar,
        )
        if not results:
            logger.warning("No information extracted")
            return

        self.extract_storage.upsert(results)
        self.extract_storage.index_done_callback()

    @async_to_sync_method
    async def generate(self, generate_config: Dict):

        batches = self.partition_storage.data
        if not batches:
            logger.warning("No partitions found for QA generation")
            return

        # Step 2： generate QA pairs
        results = await generate_qas(
            self.synthesizer_llm_client,
            batches,
            generate_config,
            progress_bar=self.progress_bar,
        )

        if not results:
            logger.warning("No QA pairs generated")
            return

        # Step 3: store the generated QA pairs
        self.qa_storage.upsert(results)
        self.qa_storage.index_done_callback()

    @async_to_sync_method
    async def clear(self):
        self.full_docs_storage.drop()
        self.chunks_storage.drop()
        self.search_storage.drop()
        self.graph_storage.clear()
        self.rephrase_storage.drop()
        self.qa_storage.drop()

        logger.info("All caches are cleared")

    # TODO: add data filtering step here in the future
    # graph_gen.filter(filter_config=config["filter"])
