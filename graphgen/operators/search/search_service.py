from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple

from graphgen.bases import BaseOperator
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger, run_concurrent

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
        data_source: str = None,
        **kwargs,
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="search"
        )
        self.data_source = data_source
        self.kwargs = kwargs
        self.search_storage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="search"
        )
        self.searcher = None

    def _init_searcher(self):
        """
        Initialize the searcher (deferred import to avoid circular imports).
        """
        if self.searcher is not None:
            return

        if not self.data_source:
            logger.error("Data source not specified")
            return

        if self.data_source == "uniprot":
            from graphgen.models import UniProtSearch

            params = self.kwargs.get("uniprot_params", {})
            self.searcher = UniProtSearch(**params)
        elif self.data_source == "ncbi":
            from graphgen.models import NCBISearch

            params = self.kwargs.get("ncbi_params", {})
            self.searcher = NCBISearch(**params)
        elif self.data_source == "rnacentral":
            from graphgen.models import RNACentralSearch

            params = self.kwargs.get("rnacentral_params", {})
            self.searcher = RNACentralSearch(**params)
        elif self.data_source == "interpro":
            from graphgen.models import InterProSearch

            params = self.kwargs.get("interpro_params", {})
            self.searcher = InterProSearch(**params)
        else:
            logger.error(f"Unknown data source: {self.data_source}")

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
            result["data_source"] = data_source
            result["type"] = seed.get("type", "text")

        return result

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Search for items in the batch using the configured data source.

        :param batch: List of items with 'content' and '_trace_id' fields
        :return: A tuple of (results, meta_updates)
            results: A list of search results.
            meta_updates: A dict mapping source IDs to lists of trace IDs for the search results.
        """
        self._init_searcher()

        if not self.searcher:
            logger.error("Searcher not initialized")
            return [], {}

        # Filter seeds with valid content and _trace_id
        seed_data = [
            item for item in batch if item and "content" in item and "_trace_id" in item
        ]

        if not seed_data:
            logger.warning("No valid seeds in batch")
            return [], {}

        # Perform concurrent searches
        results = run_concurrent(
            partial(
                self._perform_search,
                searcher_obj=self.searcher,
                data_source=self.data_source,
            ),
            seed_data,
            desc=f"Searching {self.data_source} database",
            unit="keyword",
        )

        # Filter out None results and add _trace_id from original seeds
        final_results = []
        meta_updates = {}
        for result, seed in zip(results, seed_data):
            if result is None:
                continue
            result["_trace_id"] = self.get_trace_id(result)
            final_results.append(result)
            # Map from source seed trace ID to search result trace ID
            meta_updates.setdefault(seed["_trace_id"], []).append(result["_trace_id"])

        if not final_results:
            logger.warning("No search results generated for this batch")

        return final_results, meta_updates
