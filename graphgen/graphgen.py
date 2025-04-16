# Adapt from https://github.com/HKUDS/LightRAG

import os
import asyncio
import time
from typing import List, cast, Union
from dataclasses import dataclass

from tqdm.asyncio import tqdm as tqdm_async

from models import Chunk, JsonKVStorage, OpenAIModel, NetworkXStorage, WikiSearch, Tokenizer, TraverseStrategy
from models.storage.base_storage import StorageNameSpace
from utils import create_event_loop, logger, compute_content_hash
from .operators import (extract_kg, search_wikipedia, quiz, judge_statement,
                        skip_judge_statement, traverse_graph_by_edge,
                        traverse_graph_atomically, traverse_graph_for_multi_hop)


sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@dataclass
class GraphGen:
    unique_id: int = int(time.time())
    working_dir: str = os.path.join(sys_path, "cache")

    # text chunking
    chunk_size: int = 1024
    chunk_overlap_size: int = 100

    # llm
    synthesizer_llm_client: OpenAIModel = None
    trainee_llm_client: OpenAIModel = None
    tokenizer_instance: Tokenizer = None

    # web search
    if_web_search: bool = False
    wiki_client: WikiSearch = WikiSearch()

    # traverse strategy
    traverse_strategy: TraverseStrategy = TraverseStrategy()

    def __post_init__(self):
        self.full_docs_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="full_docs"
        )
        self.text_chunks_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="text_chunks"
        )
        self.wiki_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="wiki"
        )
        self.graph_storage: NetworkXStorage = NetworkXStorage(
            self.working_dir, namespace="graph"
        )
        self.rephrase_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="rephrase"
        )
        self.qa_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", "graphgen", str(self.unique_id)), namespace=f"qa-{self.unique_id}"
        )

    async def async_split_chunks(self, data: Union[List[list], List[dict]], data_type: str) -> dict:
        # TODO： 是否进行指代消解
        if len(data) == 0:
            return {}

        new_docs = {}
        inserting_chunks = {}
        if data_type == "raw":
            assert isinstance(data, list) and isinstance(data[0], dict)
            # compute hash for each document
            new_docs = {
                compute_content_hash(doc['content'], prefix="doc-"): {'content': doc['content']} for doc in data
            }
            _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs) == 0:
                logger.warning("All docs are already in the storage")
                return {}
            logger.info("[New Docs] inserting %d docs", len(new_docs))
            for doc_key, doc in tqdm_async(
                    new_docs.items(), desc="Chunking documents", unit="doc"
                ):
                chunks = {
                    compute_content_hash(dp["content"], prefix="chunk-"): {
                        **dp,
                        'full_doc_id': doc_key
                    } for dp in self.tokenizer_instance.chunk_by_token_size(doc["content"],
                                                                            self.chunk_overlap_size, self.chunk_size)
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks_storage.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        elif data_type == "chunked":
            assert isinstance(data, list) and isinstance(data[0], list)
            new_docs = {
                compute_content_hash("".join(chunk['content']), prefix="doc-"): {'content': "".join(chunk['content'])}
                for doc in data for chunk in doc
            }
            _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs) == 0:
                logger.warning("All docs are already in the storage")
                return {}
            logger.info("[New Docs] inserting %d docs", len(new_docs))
            for doc in tqdm_async(data, desc="Chunking documents", unit="doc"):
                doc_str = "".join([chunk['content'] for chunk in doc])
                for chunk in doc:
                    chunk_key = compute_content_hash(chunk['content'], prefix="chunk-")
                    inserting_chunks[chunk_key] = {
                        **chunk,
                        'full_doc_id': compute_content_hash(doc_str, prefix="doc-")
                    }
            _add_chunk_keys = await self.text_chunks_storage.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}

        await self.full_docs_storage.upsert(new_docs)
        await self.text_chunks_storage.upsert(inserting_chunks)

        return inserting_chunks

    def insert(self, data: Union[List[list], List[dict]], data_type: str):
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert(data, data_type))

    async def async_insert(self, data: Union[List[list], List[dict]], data_type: str):
        """

        insert chunks into the graph
        """

        inserting_chunks = await self.async_split_chunks(data, data_type)

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return
        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))

        logger.info("[Entity and Relation Extraction]...")
        _add_entities_and_relations = await extract_kg(
            llm_client=self.synthesizer_llm_client,
            kg_instance=self.graph_storage,
            tokenizer_instance=self.tokenizer_instance,
            chunks=[Chunk(id=k, content=v['content']) for k, v in inserting_chunks.items()]
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted")
            return

        logger.info("[Wiki Search] is %s", 'enabled' if self.if_web_search else 'disabled')
        if self.if_web_search:
            logger.info("[Wiki Search]...")
            _add_wiki_data = await search_wikipedia(
                llm_client= self.synthesizer_llm_client,
                wiki_search_client=self.wiki_client,
                knowledge_graph_instance=_add_entities_and_relations
            )
            await self.wiki_storage.upsert(_add_wiki_data)

        await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_instance in [self.full_docs_storage, self.text_chunks_storage,
                                 self.graph_storage, self.wiki_storage]:
            if storage_instance is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_instance).index_done_callback())
        await asyncio.gather(*tasks)

    def quiz(self, max_samples=1):
        loop = create_event_loop()
        loop.run_until_complete(self.async_quiz(max_samples))

    async def async_quiz(self, max_samples=1):
        await quiz(self.synthesizer_llm_client, self.graph_storage, self.rephrase_storage, max_samples)
        await self.rephrase_storage.index_done_callback()

    def judge(self, re_judge=False, skip=False):
        loop = create_event_loop()
        loop.run_until_complete(self.async_judge(re_judge, skip))

    async def async_judge(self, re_judge=False, skip=False):
        if skip:
            _update_relations = await skip_judge_statement(self.graph_storage)
        else:
            _update_relations = await judge_statement(self.trainee_llm_client, self.graph_storage,
                                                      self.rephrase_storage, re_judge)
        await _update_relations.index_done_callback()


    def traverse(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_traverse())

    async def async_traverse(self):
        if self.traverse_strategy.qa_form == "atomic":
            results = await traverse_graph_atomically(self.synthesizer_llm_client,
                                                      self.tokenizer_instance,
                                                      self.graph_storage,
                                                      self.traverse_strategy,
                                                      self.text_chunks_storage)
        elif self.traverse_strategy.qa_form == "multi_hop":
            results = await traverse_graph_for_multi_hop(self.synthesizer_llm_client,
                                                            self.tokenizer_instance,
                                                            self.graph_storage,
                                                            self.traverse_strategy,
                                                            self.text_chunks_storage)
        else:
            results = await traverse_graph_by_edge(self.synthesizer_llm_client, self.tokenizer_instance,
                                                   self.graph_storage, self.traverse_strategy, self.text_chunks_storage)
        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()

    def clear(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_clear())

    async def async_clear(self):
        await self.full_docs_storage.drop()
        await self.text_chunks_storage.drop()
        await self.wiki_storage.drop()
        await self.graph_storage.clear()
        await self.rephrase_storage.drop()
        await self.qa_storage.drop()

        logger.info("All caches are cleared")
