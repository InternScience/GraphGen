"""
To use Google Web Search API,
follow the instructions [here](https://developers.google.com/custom-search/v1/overview)
to get your Google searcher api key.

To use Bing Web Search API,
follow the instructions [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
and obtain your Bing subscription key.
"""

import numpy as np
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

        # 初始化，根据data_sources选择
        self.searchers = {
            "uniprot": UniProtSearcher(**kwargs),
            "ncbi": NCBISearcher(**kwargs),
            "rnacentral": RNAcentralSearcher(**kwargs),
        }

    def _perform_searches(self, seed_data: list) -> dict:
        """
        Internal method to perform searches across multiple search types and aggregate the results.
        :param seed_data: A list of seed data dictionaries to search for
        :return: A dictionary with search results
        """
        results = {}

        for data_source in self.data_sources:
            # # Prepare save callback for this data source
            # def make_save_callback(source_name):
            #     def save_callback(intermediate_results, completed_count):
            #         """Save intermediate search results."""
            #         if self.search_storage is None:
            #             return
            #
            #         # Convert results list to dict format
            #         # Results are tuples of (query, result_dict) or just result_dict
            #         batch_results = {}
            #         for result in intermediate_results:
            #             if result is None:
            #                 continue
            #             # Check if result is a dict with _search_query key
            #             if isinstance(result, dict) and "_search_query" in result:
            #                 query = result["_search_query"]
            #                 # Create a key for the result (using query as key)
            #                 key = f"{source_name}:{query}"
            #                 batch_results[key] = result
            #             elif isinstance(result, dict):
            #                 # If no _search_query, use a generated key
            #                 key = f"{source_name}:{completed_count}"
            #                 batch_results[key] = result
            #
            #         if batch_results:
            #             # Filter out already existing keys
            #             new_keys = self.search_storage.filter_keys(list(batch_results.keys()))
            #             new_results = {k: v for k, v in batch_results.items() if k in new_keys}
            #             if new_results:
            #                 self.search_storage.upsert(new_results)
            #                 self.search_storage.index_done_callback()
            #                 self.logger.debug("Saved %d intermediate results for %s", len(new_results), source_name)
            #
            #     return save_callback

            if data_source == "uniprot":
                from graphgen.models import UniProtSearch

                uniprot_params = self.kwargs.get("uniprot_params", {})
                # searcher = UniProtSearch(working_dir=self.working_dir, **uniprot_params)
                searcher = self.searchers["uniprot"]

            elif data_source == "ncbi":
                from graphgen.models import NCBISearch

                ncbi_params = self.kwargs.get("ncbi_params", {})
                searcher = NCBISearch(working_dir=self.working_dir, **ncbi_params)

            elif data_source == "rnacentral":
                from graphgen.models import RNACentralSearch

                rnacentral_params = self.kwargs.get("rnacentral_params", {})

                searcher = RNACentralSearch(
                    working_dir=self.working_dir, **rnacentral_params
                )

            elif data_source == "google":
                # TODO: Implement Google searcher here
                continue
            elif data_source == "bing":
                # TODO: Implement Bing searcher here
                continue
            elif data_source == "wikipedia":
                # TODO: Implement Wikipedia searcher here
                continue
            else:
                self.logger.error("Data source %s not supported.", data_source)
                continue

            # 3 如果search_result中有有重复的，直接跳过
            # key value
            # key: datasource-compute_content_hash(query)

            for seed in seed_data:
                query = seed["_search_query"]
                key = f"{data_source}-{compute_content_hash(query)}"
                if key in self.search_storage:
                    self.logger.info("Duplicate query found: %s", query)
                    continue

            search_results = run_concurrent(
                searcher.search,
                seed_data,
                desc=f"Searching {data_source} database",
                unit="keyword",
            )
            # results[data_source] = search_results
            # 可以是key value的格式

        return results

    @staticmethod
    def _clean_value(v):
        """Recursively convert numpy arrays and other problematic types to Python-native types."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [SearchService._clean_value(item) for item in v]
        if isinstance(v, dict):
            return {k: SearchService._clean_value(val) for k, val in v.items()}
        return v

    def _normalize_searched_data(
        self, doc: dict
    ) -> dict:  # pylint: disable=too-many-branches
        """
        Normalize a document that already contains search results to the expected format.

        :param doc: Document dictionary with search results
        :return: Normalized document dictionary
        """
        # Ensure required fields exist
        doc_id = doc.get("_doc_id")
        if not doc_id:
            # Generate doc_id from id or other fields
            raw_doc_id = (
                doc.get("id") or doc.get("_search_query") or f"doc-{hash(str(doc))}"
            )
            doc_id = str(raw_doc_id)

        # Ensure doc_id starts with "doc-" prefix
        if not doc_id.startswith("doc-"):
            doc_id = f"doc-{doc_id}"

        # Determine document type from molecule_type or existing type
        doc_type = doc.get("type", "text")
        if doc_type == "text" and "molecule_type" in doc:
            molecule_type = doc.get("molecule_type", "").lower()
            if molecule_type in ["dna", "rna", "protein"]:
                doc_type = molecule_type

        # Ensure data_source field exists
        data_source = doc.get("data_source")
        if not data_source:
            # Infer from database field
            database = doc.get("database", "").lower()
            if "uniprot" in database:
                data_source = "uniprot"
            elif "ncbi" in database:
                data_source = "ncbi"
            elif "rnacentral" in database or "rna" in database:
                data_source = "rnacentral"

        # Build or preserve content field
        content = doc.get("content")
        if not content or content.strip() == "":
            # Build content from available fields if missing
            content_parts = []
            if doc.get("title"):
                content_parts.append(f"Title: {doc['title']}")
            if doc.get("description"):
                content_parts.append(f"Description: {doc['description']}")
            if doc.get("function"):
                func = doc["function"]
                if isinstance(func, list):
                    func = ", ".join(str(f) for f in func)
                content_parts.append(f"Function: {func}")
            if doc.get("sequence"):
                content_parts.append(f"Sequence: {doc['sequence']}")

            if not content_parts:
                # Fallback: create content from key fields
                key_fields = [
                    "protein_name",
                    "gene_name",
                    "gene_description",
                    "organism",
                ]
                for field in key_fields:
                    if field in doc and doc[field]:
                        content_parts.append(f"{field}: {doc[field]}")

            content = "\n".join(content_parts) if content_parts else str(doc)

        # Create normalized row
        normalized_doc = {
            "_doc_id": doc_id,
            "type": doc_type,
            "content": content,
            "data_source": data_source,
            **doc,  # Include all original fields for metadata
        }

        return normalized_doc

    def process(
        self, batch: pd.DataFrame
    ) -> pd.DataFrame:  # pylint: disable=too-many-branches
        """
        Process a batch of documents and perform searches.
        This is the Ray Data operator interface.

        If input data already contains search results (detected by presence of
        data_source, database, or search-specific fields), the search step is
        skipped and the data is normalized and returned directly.

        :param batch: DataFrame containing documents with at least '_doc_id' and 'content' columns
        :return: DataFrame containing search results
        """
        docs = batch.to_dict(orient="records")

        # Data doesn't contain search results, perform search as usual
        # seed_data = {doc.get("_doc_id", f"doc-{i}"): doc for i, doc in enumerate(docs)}

        # docs may contain None entries, filter them out & remove duplicates based on content
        # 1 去重
        unique_contents = set()
        seed_data = []
        for doc in docs:
            if not doc or "content" not in doc:
                continue
            content = doc["content"]
            if content not in unique_contents:
                unique_contents.add(content)
                seed_data.append(doc)

        search_results = self._perform_searches(seed_data)

        # 4 更新到search_storage里

        # query json dict

        # Convert search_results from {data_source: [results]} to DataFrame
        # Each result becomes a document row compatible with chunk service
        result_rows = []

        # for data_source, result_list in search_results.items():
        #     if not isinstance(result_list, list):
        #         continue

        #     for result in result_list:
        #         if result is None:
        #             continue

        #         # Convert search result to document format expected by chunk service
        #         # Build content from available fields
        #         content_parts = []
        #         if result.get("title"):
        #             content_parts.append(f"Title: {result['title']}")
        #         if result.get("description"):
        #             content_parts.append(f"Description: {result['description']}")
        #         if result.get("function"):
        #             content_parts.append(f"Function: {result['function']}")
        #         if result.get("sequence"):
        #             content_parts.append(f"Sequence: {result['sequence']}")

        #         # If no content parts, use a default or combine all fields
        #         if not content_parts:
        #             # Fallback: create content from all string fields
        #             content_parts = [
        #                 f"{k}: {v}"
        #                 for k, v in result.items()
        #                 if isinstance(v, (str, int, float)) and k != "_search_query"
        #             ]

        #         content = "\n".join(content_parts) if content_parts else str(result)

        #         # Determine document type from molecule_type or default to "text"
        #         doc_type = result.get("molecule_type", "text").lower()
        #         if doc_type not in ["text", "dna", "rna", "protein"]:
        #             doc_type = "text"

        #         # Convert to string to handle Ray Data ListElement and other types
        #         raw_doc_id = (
        #             result.get("id")
        #             or result.get("_search_query")
        #             or f"search-{len(result_rows)}"
        #         )
        #         doc_id = str(raw_doc_id)

        #         # Ensure doc_id starts with "doc-" prefix
        #         if not doc_id.startswith("doc-"):
        #             doc_id = f"doc-{doc_id}"

                # Convert numpy arrays and complex types to Python-native types
                # to avoid Ray Data tensor extension casting issues
                # cleaned_result = {k: self._clean_value(v) for k, v in result.items()}

        # TODO: 待定
                # Create document row with all result fields plus required fields
                row = {
                    "_doc_id": doc_id,
                    "type": doc_type,
                    "content": content,
                    "data_source": data_source,
                    **cleaned_result,  # Include all original result fields for metadata
                }
                result_rows.append(row)

        if not result_rows:
            self.logger.warning("No search results generated for this batch")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=["_doc_id", "type", "content", "data_source"])

        return pd.DataFrame(result_rows)
