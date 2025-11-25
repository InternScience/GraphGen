"""
To use Google Web Search API,
follow the instructions [here](https://developers.google.com/custom-search/v1/overview)
to get your Google searcher api key.

To use Bing Web Search API,
follow the instructions [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
and obtain your Bing subscription key.
"""


from graphgen.utils import logger, run_concurrent


async def search_all(
    seed_data: dict,
    search_config: dict,
) -> dict:
    """
    Perform searches across multiple search types and aggregate the results.
    :param seed_data: A dictionary containing seed data with entity names.
    :param search_config: A dictionary specifying which data sources to use for searching.
    :return: A dictionary with doc_hash as keys and search results as values.
    """

    results = {}
    data_sources = search_config.get("data_sources", [])

    for data_source in data_sources:
        if data_source == "uniprot":
            from graphgen.models import UniProtSearch

            uniprot_search_client = UniProtSearch(
                **search_config.get("uniprot_params", {})
            )

            # Prepare search queries: map doc_hash to content
            doc_queries = {}
            for doc_hash, doc_data in seed_data.items():
                # Try to extract search query from different data types
                query = None
                if "content" in doc_data:
                    query = doc_data["content"]
                elif doc_data.get("type") == "protein" and "protein_caption" in doc_data:
                    # For protein type, try to use sequence, id, or protein_name
                    protein_caption = doc_data["protein_caption"]
                    if "sequence" in protein_caption and protein_caption["sequence"]:
                        query = protein_caption["sequence"]
                    elif "id" in protein_caption and protein_caption["id"]:
                        query = protein_caption["id"]
                    elif "protein_name" in protein_caption and protein_caption["protein_name"]:
                        query = protein_caption["protein_name"]
                
                if query:
                    if query not in doc_queries:
                        doc_queries[query] = []
                    doc_queries[query].append(doc_hash)

            # Get unique queries
            unique_queries = list(doc_queries.keys())
            
            # Perform searches
            uniprot_results = await run_concurrent(
                uniprot_search_client.search,
                unique_queries,
                desc="Searching UniProt database",
                unit="keyword",
            )

            # Map results back to doc hashes
            for query, result in zip(unique_queries, uniprot_results):
                for doc_hash in doc_queries[query]:
                    if doc_hash not in results:
                        results[doc_hash] = {}
                    results[doc_hash][data_source] = result
        else:
            logger.error("Data source %s not supported.", data_source)
            continue

    return results
