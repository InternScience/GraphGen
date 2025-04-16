import asyncio
from graphgen.models import WikiSearch, OpenAIModel
from graphgen.models.storage.base_storage import BaseGraphStorage
from graphgen.templates import SEARCH_JUDGEMENT_PROMPT
from graphgen.utils import logger


async def _process_single_entity(entity_name: str,
                                 description: str,
                                 llm_client: OpenAIModel,
                                 wiki_search_client: WikiSearch) -> tuple[str, None] | tuple[str, str]:
    """
    Process single entity

    """
    search_results = await wiki_search_client.search(entity_name)
    if not search_results:
        return entity_name, None
    examples = "\n".join(SEARCH_JUDGEMENT_PROMPT["EXAMPLES"])
    search_results.append("None of the above")

    search_results_str = "\n".join([f"{i + 1}. {sr}" for i, sr in enumerate(search_results)])
    prompt = SEARCH_JUDGEMENT_PROMPT["TEMPLATE"].format(
        examples=examples,
        entity_name=entity_name,
        description=description,
        search_results=search_results_str,
    )
    response = await llm_client.generate_answer(prompt)

    try:
        response = response.strip()
        response = int(response)
        if response < 1 or response >= len(search_results):
            response = None
        else:
            response = await wiki_search_client.summary(search_results[response - 1])
    except ValueError:
        response = None

    logger.info("Entity %s search result: %s response: %s", entity_name, str(search_results), response)

    return entity_name, response

async def search_wikipedia(llm_client: OpenAIModel,
                           wiki_search_client: WikiSearch,
                           knowledge_graph_instance: BaseGraphStorage,) -> dict:
    """
    Search wikipedia for entities

    :param llm_client: LLM model
    :param wiki_search_client: wiki search client
    :param knowledge_graph_instance: knowledge graph instance
    :return: nodes with search results
    """


    nodes = await knowledge_graph_instance.get_all_nodes()
    nodes = list(nodes)
    wiki_data = {}

    tasks = [
        _process_single_entity(node[0].strip('"'), node[1]["description"], llm_client, wiki_search_client)
        for node in nodes
    ]

    for task in asyncio.as_completed(tasks):
        result = await task
        wiki_data[result[0]] = result[1]

    return wiki_data
