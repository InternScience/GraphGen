"""
To use Google Web Search API,
follow the instructions [here](https://developers.google.com/custom-search/v1/overview)
to get your Google searcher api key.

To use Bing Web Search API,
follow the instructions [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
and obtain your Bing subscription key.
"""

import pandas as pd

from graphgen.bases import BaseOperator
from graphgen.common import init_storage
from graphgen.utils import run_concurrent


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

        # 初始化所有 searchers（延迟导入以避免循环导入）
        from graphgen.models import NCBISearch, RNACentralSearch, UniProtSearch

        uniprot_params = kwargs.get("uniprot_params", {})
        ncbi_params = kwargs.get("ncbi_params", {})
        rnacentral_params = kwargs.get("rnacentral_params", {})

        self.searchers = {
            "uniprot": UniProtSearch(working_dir=self.working_dir, **uniprot_params),
            "ncbi": NCBISearch(working_dir=self.working_dir, **ncbi_params),
            "rnacentral": RNACentralSearch(working_dir=self.working_dir, **rnacentral_params),
        }

    def _perform_searches(self, seed_data: list) -> dict:
        """
        Internal method to perform searches across multiple search types and aggregate the results.
        :param seed_data: A list of seed data dictionaries to search for
        :return: A dictionary with search results
        """
        results = {}

        for data_source in self.data_sources:
            if data_source not in self.searchers:
                if data_source in ["google", "bing", "wikipedia"]:
                    # TODO: Implement these searchers here
                    continue
                self.logger.error("Data source %s not supported.", data_source)
                continue

            searcher = self.searchers[data_source]

            # 创建异步包装器，将同步的search方法包装成异步
            async def async_search_wrapper(seed: dict, searcher_obj=searcher, ds=data_source):
                import asyncio
                query = seed.get("_search_query") or seed.get("content", "")
                threshold = seed.get("threshold", 0.01)
                # 在executor中运行同步的search方法
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, searcher_obj.search, query, threshold)
                if result:
                    # 生成 _doc_id（从 id 字段，确保以 "doc-" 开头）
                    doc_id = result.get("id") or result.get("_search_query") or f"doc-{hash(str(result))}"
                    doc_id = str(doc_id)
                    if not doc_id.startswith("doc-"):
                        doc_id = f"doc-{doc_id}"
                    result["_doc_id"] = doc_id

                    # 直接添加已知的 data_source
                    result["data_source"] = ds

                    # 设置 type 字段（从输入数据获取，如果没有则默认为 "text"）
                    if "type" in seed:
                        result["type"] = seed.get("type", "text")
                    else:
                        result["type"] = "text"
                return result

            search_results = run_concurrent(
                async_search_wrapper,
                seed_data,
                desc=f"Searching {data_source} database",
                unit="keyword",
            )
            results[data_source] = search_results

        return results

    def process(
        self, batch: pd.DataFrame
    ) -> pd.DataFrame:  # pylint: disable=too-many-branches
        """
        Process a batch of documents and perform searches.
        This is the Ray Data operator interface.

        :param batch: DataFrame containing documents with at least 'content' column
        :return: DataFrame containing search results with '_doc_id', 'type', 'data_source' fields
        """
        docs = batch.to_dict(orient="records")

        # Filter out None entries and documents without content
        seed_data = [doc for doc in docs if doc and "content" in doc]

        search_results = self._perform_searches(seed_data)

        # Convert search_results from {data_source: [results]} to DataFrame
        result_rows = []

        for result_list in search_results.values():
            if not isinstance(result_list, list):
                continue

            for result in result_list:
                if result is not None:
                    result_rows.append(result)

        if not result_rows:
            self.logger.warning("No search results generated for this batch")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=["_doc_id", "type", "content", "data_source"])

        return pd.DataFrame(result_rows)
