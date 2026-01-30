from functools import partial
from typing import TYPE_CHECKING, Optional

from graphgen.bases import BaseOperator
from graphgen.common import init_storage
from graphgen.utils import compute_content_hash, logger, run_concurrent

if TYPE_CHECKING:
    import pandas as pd


class SearchService(BaseOperator):
    """
    Service class for performing searches across multiple data sources.
    Provides search functionality for UniProt, NCBI, and RNAcentral databases.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        data_sources: list = None,
        **kwargs,
    ):
        super().__init__(working_dir=working_dir, op_name="search_service")
        self.working_dir = working_dir
        self.data_sources = data_sources or []
        self.kwargs = kwargs
        self.search_storage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="search"
        )
        self.searchers = {}

    def _init_searchers(self):
        """
        Initialize all searchers (deferred import to avoid circular imports).
        """
        for datasource in self.data_sources:
            if datasource in self.searchers:
                continue
            if datasource == "uniprot":
                from graphgen.models import UniProtSearch

                params = self.kwargs.get("uniprot_params", {})
                self.searchers[datasource] = UniProtSearch(**params)
            elif datasource == "ncbi":
                from graphgen.models import NCBISearch

                params = self.kwargs.get("ncbi_params", {})
                self.searchers[datasource] = NCBISearch(**params)
            elif datasource == "rnacentral":
                from graphgen.models import RNACentralSearch

                params = self.kwargs.get("rnacentral_params", {})
                self.searchers[datasource] = RNACentralSearch(**params)
            else:
                logger.error(f"Unknown data source: {datasource}, skipping")

    @staticmethod
    async def _perform_search(
        seed: dict, searcher_obj, data_source: str
    ) -> Optional[dict]:
        """
        Perform search for a single seed using the specified searcher.

        :param seed: The seed document with 'content' field
        :param searcher_obj: The searcher instance
        :param data_source: The data source name
        :return: Search result with metadata
        """
        query = seed.get("content", "")

        if not query:
            logger.warning("Empty query for seed: %s", seed)
            return None

        result = searcher_obj.search(query)
        if result:
            result["_doc_id"] = compute_content_hash(str(data_source) + query, "doc-")
            result["data_source"] = data_source
            result["type"] = seed.get("type", "text")

        return result

    def _process_single_source(
        self, data_source: str, seed_data: list[dict]
    ) -> list[dict]:
        """
        process a single data source: check cache, search missing, update cache.
        """
        searcher = self.searchers[data_source]

        seeds_with_ids = []
        for seed in seed_data:
            query = seed.get("content", "")
            if not query:
                continue
            doc_id = compute_content_hash(str(data_source) + query, "doc-")
            seeds_with_ids.append((doc_id, seed))

        if not seeds_with_ids:
            return []

        doc_ids = [doc_id for doc_id, _ in seeds_with_ids]
        cached_results = self.search_storage.get_by_ids(doc_ids)

        to_search_seeds = []
        final_results = []

        for (doc_id, seed), cached in zip(seeds_with_ids, cached_results):
            if cached is not None:
                if "_doc_id" not in cached:
                    cached["_doc_id"] = doc_id
                final_results.append(cached)
            else:
                to_search_seeds.append(seed)

        if to_search_seeds:
            new_results = run_concurrent(
                partial(
                    self._perform_search, searcher_obj=searcher, data_source=data_source
                ),
                to_search_seeds,
                desc=f"Searching {data_source} database",
                unit="keyword",
            )
            new_results = [res for res in new_results if res is not None]

            if new_results:
                upsert_data = {res["_doc_id"]: res for res in new_results}
                self.search_storage.upsert(upsert_data)
                logger.info(
                    f"Saved {len(upsert_data)} new results to {data_source} cache"
                )

            final_results.extend(new_results)

        return final_results

    def process(self, batch: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        docs = batch.to_dict(orient="records")

        self._init_searchers()

        seed_data = [doc for doc in docs if doc and "content" in doc]

        if not seed_data:
            logger.warning("No valid seeds in batch")
            return pd.DataFrame([])

        all_results = []

        for data_source in self.data_sources:
            if data_source not in self.searchers:
                logger.error(f"Data source {data_source} not initialized, skipping")
                continue

            source_results = self._process_single_source(data_source, seed_data)
            all_results.extend(source_results)

        if not all_results:
            logger.warning("No search results generated for this batch")

        return pd.DataFrame(all_results)
