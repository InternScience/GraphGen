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

    def _is_already_searched(self, doc: dict) -> bool:
        """
        Check if a document already contains search results.
        
        :param doc: Document dictionary
        :return: True if document appears to already contain search results
        """
        # Check for data_source field (added by search_service)
        if "data_source" in doc and doc["data_source"]:
            return True
        
        # Check for database field (added by search operations)
        if "database" in doc and doc["database"] in ["UniProt", "NCBI", "RNAcentral"]:
            # Also check for molecule_type to confirm it's a search result
            if "molecule_type" in doc and doc["molecule_type"] in ["DNA", "RNA", "protein"]:
                return True
        
        # Check for search-specific fields that indicate search results
        search_indicators = [
            "uniprot_id", "entry_name",  # UniProt
            "gene_id", "gene_name", "chromosome",  # NCBI
            "rnacentral_id", "rna_type",  # RNAcentral
        ]
        if any(indicator in doc for indicator in search_indicators):
            # Make sure it's not just metadata by checking for database or molecule_type
            if "database" in doc or "molecule_type" in doc:
                return True
        
        return False

    def _normalize_searched_data(self, doc: dict) -> dict:
        """
        Normalize a document that already contains search results to the expected format.
        
        :param doc: Document dictionary with search results
        :return: Normalized document dictionary
        """
        # Ensure required fields exist
        doc_id = doc.get("_doc_id")
        if not doc_id:
            # Generate doc_id from id or other fields
            raw_doc_id = doc.get("id") or doc.get("_search_query") or f"doc-{hash(str(doc))}"
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
                key_fields = ["protein_name", "gene_name", "gene_description", "organism"]
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

    def process(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of documents and perform searches.
        This is the Ray Data operator interface.
        
        If input data already contains search results (detected by presence of 
        data_source, database, or search-specific fields), the search step is 
        skipped and the data is normalized and returned directly.

        :param batch: DataFrame containing documents with at least '_doc_id' and 'content' columns
        :return: DataFrame containing search results
        """
        # Convert DataFrame to dictionary format
        docs = batch.to_dict(orient="records")
        
        # Check if data already contains search results
        already_searched = all(self._is_already_searched(doc) for doc in docs if doc)
        
        if already_searched:
            # Data already contains search results, normalize and return directly
            self.logger.info(
                "Input data already contains search results. Skipping search step and normalizing data."
            )
            result_rows = []
            for doc in docs:
                if not doc:
                    continue
                normalized_doc = self._normalize_searched_data(doc)
                result_rows.append(normalized_doc)
            
            if not result_rows:
                self.logger.warning("No documents found in batch")
                return pd.DataFrame(columns=["_doc_id", "type", "content", "data_source"])
            
            return pd.DataFrame(result_rows)
        
        # Data doesn't contain search results, perform search as usual
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
                
                # Convert to string to handle Ray Data ListElement and other types
                raw_doc_id = result.get("id") or result.get("_search_query") or f"search-{len(result_rows)}"
                doc_id = str(raw_doc_id)
                
                # Ensure doc_id starts with "doc-" prefix
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
