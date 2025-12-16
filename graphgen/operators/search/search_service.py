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
from graphgen.utils import create_event_loop, run_concurrent


class SearchService(BaseOperator):
    """
    Service class for performing searches across multiple data sources.
    Provides search functionality for UniProt, NCBI, and RNAcentral databases.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        data_sources: list = None,
        ncbi_params: dict = None,
        uniprot_params: dict = None,
        rnacentral_params: dict = None,
        save_interval: int = 1000,
        **kwargs,
    ):
        super().__init__(working_dir=working_dir, op_name="search_service")
        self.working_dir = working_dir
        
        # Build search_config dictionary from parameters
        self.search_config = {
            "data_sources": data_sources or [],
        }
        
        if ncbi_params:
            self.search_config["ncbi_params"] = ncbi_params
        if uniprot_params:
            self.search_config["uniprot_params"] = uniprot_params
        if rnacentral_params:
            self.search_config["rnacentral_params"] = rnacentral_params
        
        self.save_interval = save_interval
        self.search_storage = None  # Optional: can be initialized if needed for saving intermediate results

    async def _perform_searches(self, seed_data: dict) -> dict:
        """
        Internal method to perform searches across multiple search types and aggregate the results.
        :param seed_data: A dictionary containing seed data with entity names.
        :return: A dictionary with search results
        """
        results = {}
        data_sources = self.search_config.get("data_sources", [])

        for data_source in data_sources:
            data = list(seed_data.values())
            data = [d["content"] for d in data if "content" in d]
            data = list(set(data))  # Remove duplicates

            # Prepare save callback for this data source
            def make_save_callback(source_name):
                def save_callback(intermediate_results, completed_count):
                    """Save intermediate search results."""
                    if self.search_storage is None:
                        return
                    
                    # Convert results list to dict format
                    # Results are tuples of (query, result_dict) or just result_dict
                    batch_results = {}
                    for result in intermediate_results:
                        if result is None:
                            continue
                        # Check if result is a dict with _search_query key
                        if isinstance(result, dict) and "_search_query" in result:
                            query = result["_search_query"]
                            # Create a key for the result (using query as key)
                            key = f"{source_name}:{query}"
                            batch_results[key] = result
                        elif isinstance(result, dict):
                            # If no _search_query, use a generated key
                            key = f"{source_name}:{completed_count}"
                            batch_results[key] = result
                    
                    if batch_results:
                        # Filter out already existing keys
                        new_keys = self.search_storage.filter_keys(list(batch_results.keys()))
                        new_results = {k: v for k, v in batch_results.items() if k in new_keys}
                        if new_results:
                            self.search_storage.upsert(new_results)
                            self.search_storage.index_done_callback()
                            self.logger.debug("Saved %d intermediate results for %s", len(new_results), source_name)
                
                return save_callback

            if data_source == "uniprot":
                from graphgen.models import UniProtSearch

                uniprot_params = self.search_config.get("uniprot_params", {}).copy()
                # Get max_concurrent from config before passing params to constructor
                max_concurrent = uniprot_params.pop("max_concurrent", None)
                
                uniprot_search_client = UniProtSearch(
                    working_dir=self.working_dir,
                    **uniprot_params
                )

                uniprot_results = await run_concurrent(
                    uniprot_search_client.search,
                    data,
                    desc="Searching UniProt database",
                    unit="keyword",
                    save_interval=self.save_interval if self.save_interval > 0 else 0,
                    save_callback=make_save_callback("uniprot") if self.search_storage and self.save_interval > 0 else None,
                    max_concurrent=max_concurrent,
                )
                results[data_source] = uniprot_results

            elif data_source == "ncbi":
                from graphgen.models import NCBISearch

                ncbi_params = self.search_config.get("ncbi_params", {}).copy()
                # Get max_concurrent from config before passing params to constructor
                max_concurrent = ncbi_params.pop("max_concurrent", None)
                
                ncbi_search_client = NCBISearch(
                    working_dir=self.working_dir,
                    **ncbi_params
                )

                ncbi_results = await run_concurrent(
                    ncbi_search_client.search,
                    data,
                    desc="Searching NCBI database",
                    unit="keyword",
                    save_interval=self.save_interval if self.save_interval > 0 else 0,
                    save_callback=make_save_callback("ncbi") if self.search_storage and self.save_interval > 0 else None,
                    max_concurrent=max_concurrent,
                )
                results[data_source] = ncbi_results

            elif data_source == "rnacentral":
                from graphgen.models import RNACentralSearch

                rnacentral_params = self.search_config.get("rnacentral_params", {}).copy()
                # Get max_concurrent from config before passing params to constructor
                max_concurrent = rnacentral_params.pop("max_concurrent", None)
                
                rnacentral_search_client = RNACentralSearch(
                    working_dir=self.working_dir,
                    **rnacentral_params
                )

                rnacentral_results = await run_concurrent(
                    rnacentral_search_client.search,
                    data,
                    desc="Searching RNAcentral database",
                    unit="keyword",
                    save_interval=self.save_interval if self.save_interval > 0 else 0,
                    save_callback=make_save_callback("rnacentral") if self.search_storage and self.save_interval > 0 else None,
                    max_concurrent=max_concurrent,
                )
                results[data_source] = rnacentral_results

            else:
                self.logger.error("Data source %s not supported.", data_source)
                continue

        return results

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of documents and perform searches.
        This is the Ray Data operator interface.

        :param batch: DataFrame containing documents with at least '_doc_id' and 'content' columns
        :return: DataFrame containing search results
        """
        # Convert DataFrame to dictionary format
        docs = batch.to_dict(orient="records")
        seed_data = {doc.get("_doc_id", f"doc-{i}"): doc for i, doc in enumerate(docs)}
        
        # Perform searches asynchronously
        loop, created = create_event_loop()
        try:
            if loop.is_running():
                # If loop is already running, we can't use run_until_complete
                # This shouldn't happen in normal usage, but handle it gracefully
                raise RuntimeError(
                    "Cannot use process when event loop is already running. "
                    "This is likely a Ray worker configuration issue."
                )
            search_results = loop.run_until_complete(
                self._perform_searches(seed_data)
            )
        finally:
            # Only close the loop if we created it
            if created:
                loop.close()
        
        # Convert search_results from {data_source: [results]} to DataFrame
        # Each result becomes a document row compatible with chunk service
        result_rows = []
        
        for data_source, result_list in search_results.items():
            if not isinstance(result_list, list):
                continue
            
            for result in result_list:
                if result is None:
                    continue
                
                # Convert search result to document format expected by chunk service
                # Build content from available fields
                content_parts = []
                if result.get("title"):
                    content_parts.append(f"Title: {result['title']}")
                if result.get("description"):
                    content_parts.append(f"Description: {result['description']}")
                if result.get("function"):
                    content_parts.append(f"Function: {result['function']}")
                if result.get("sequence"):
                    content_parts.append(f"Sequence: {result['sequence']}")
                
                # If no content parts, use a default or combine all fields
                if not content_parts:
                    # Fallback: create content from all string fields
                    content_parts = [
                        f"{k}: {v}" 
                        for k, v in result.items() 
                        if isinstance(v, (str, int, float)) and k != "_search_query"
                    ]
                
                content = "\n".join(content_parts) if content_parts else str(result)
                
                # Determine document type from molecule_type or default to "text"
                doc_type = result.get("molecule_type", "text").lower()
                if doc_type not in ["text", "dna", "rna", "protein"]:
                    doc_type = "text"
                
                # Generate document ID from result ID or search query
                doc_id = result.get("id") or result.get("_search_query") or f"search-{len(result_rows)}"
                if not doc_id.startswith("doc-"):
                    doc_id = f"doc-{doc_id}"
                
                # Create document row with all result fields plus required fields
                row = {
                    "_doc_id": doc_id,
                    "type": doc_type,
                    "content": content,
                    "data_source": data_source,
                    **result,  # Include all original result fields for metadata
                }
                result_rows.append(row)
        
        if not result_rows:
            self.logger.warning("No search results generated for this batch")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=["_doc_id", "type", "content", "data_source"])
        
        return pd.DataFrame(result_rows)
