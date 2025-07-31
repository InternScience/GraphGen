from graphgen.models import NetworkXStorage, OpenAIModel
from graphgen.utils import logger


async def search_all(
    llm_client: OpenAIModel, search_types: dict, kg_instance: NetworkXStorage
) -> dict[str, dict[str, str]]:
    """
    :param llm_client
    :param search_types
    :param kg_instance
    :return: nodes with search results
    """

    # 增量建图时，只需要搜索新增实体

    results = {}

    for search_type in search_types:
        if search_type == "wikipedia":
            from graphgen.models import WikiSearch
            from graphgen.operators.search.kg.search_wikipedia import search_wikipedia

            wiki_search_client = WikiSearch()

            wiki_results = await search_wikipedia(
                llm_client, wiki_search_client, kg_instance
            )
            for entity_name, description in wiki_results.items():
                if description:
                    results[entity_name] = {"wikipedia": description}
        # elif search_type == "google":
        #     from graphgen.operators.search.web.search_google import search_google
        #     return await search_google(llm_client, kg_instance)
        #
        # elif search_type == "bing":
        #     from graphgen.operators.search.web.search_bing import search_bing
        #     return await search_bing(llm_client, kg_instance)

        else:
            logger.error("Search type %s is not supported yet.", search_type)
            continue

    return results
