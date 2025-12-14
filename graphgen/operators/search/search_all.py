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
    search_storage=None,
    save_interval: int = 1000,
) -> dict:
    """
    Perform searches across multiple search types and aggregate the results.
    :param seed_data: A dictionary containing seed data with entity names.
    :param search_config: A dictionary specifying which data sources to use for searching.
    :param search_storage: Optional storage instance for periodic saving of results.
    :param save_interval: Number of search results to accumulate before saving (default: 1000, 0 to disable).
    :return: A dictionary with search results
    """

    results = {}
    data_sources = search_config.get("data_sources", [])

    for data_source in data_sources:
        data = list(seed_data.values())
        data = [d["content"] for d in data if "content" in d]
        data = list(set(data))  # Remove duplicates

        # Prepare save callback for this data source
        def make_save_callback(source_name):
            def save_callback(intermediate_results, completed_count):
                """Save intermediate search results."""
                if search_storage is None:
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
                    new_keys = search_storage.filter_keys(list(batch_results.keys()))
                    new_results = {k: v for k, v in batch_results.items() if k in new_keys}
                    if new_results:
                        search_storage.upsert(new_results)
                        search_storage.index_done_callback()
                        logger.debug("Saved %d intermediate results for %s", len(new_results), source_name)
            
            return save_callback

        if data_source == "uniprot":
            from graphgen.models import UniProtSearch

            uniprot_params = search_config.get("uniprot_params", {})
            uniprot_search_client = UniProtSearch(
                **uniprot_params
            )
            
            # Get max_concurrent from config, default to None (unlimited) for backward compatibility
            max_concurrent = uniprot_params.get("max_concurrent")

            uniprot_results = await run_concurrent(
                uniprot_search_client.search,
                data,
                desc="Searching UniProt database",
                unit="keyword",
                save_interval=save_interval if save_interval > 0 else 0,
                save_callback=make_save_callback("uniprot") if search_storage and save_interval > 0 else None,
                max_concurrent=max_concurrent,
            )
            results[data_source] = uniprot_results

        elif data_source == "ncbi":
            from graphgen.models import NCBISearch

            ncbi_params = search_config.get("ncbi_params", {})
            ncbi_search_client = NCBISearch(
                **ncbi_params
            )
            
            # Get max_concurrent from config, default to None (unlimited) for backward compatibility
            max_concurrent = ncbi_params.get("max_concurrent")

            ncbi_results = await run_concurrent(
                ncbi_search_client.search,
                data,
                desc="Searching NCBI database",
                unit="keyword",
                save_interval=save_interval if save_interval > 0 else 0,
                save_callback=make_save_callback("ncbi") if search_storage and save_interval > 0 else None,
                max_concurrent=max_concurrent,
            )
            results[data_source] = ncbi_results

        elif data_source == "rnacentral":
            from graphgen.models import RNACentralSearch

            rnacentral_params = search_config.get("rnacentral_params", {})
            rnacentral_search_client = RNACentralSearch(
                **rnacentral_params
            )
            
            # Get max_concurrent from config, default to None (unlimited) for backward compatibility
            max_concurrent = rnacentral_params.get("max_concurrent")

            rnacentral_results = await run_concurrent(
                rnacentral_search_client.search,
                data,
                desc="Searching RNAcentral database",
                unit="keyword",
                save_interval=save_interval if save_interval > 0 else 0,
                save_callback=make_save_callback("rnacentral") if search_storage and save_interval > 0 else None,
                max_concurrent=max_concurrent,
            )
            results[data_source] = rnacentral_results

        else:
            logger.error("Data source %s not supported.", data_source)
            continue

    return results
