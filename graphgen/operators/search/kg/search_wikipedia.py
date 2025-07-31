from tqdm.asyncio import tqdm_asyncio as tqdm_async

from graphgen.models import NetworkXStorage, OpenAIModel, WikiSearch
from graphgen.templates import SEARCH_JUDGEMENT_PROMPT
from graphgen.utils import logger


async def _process_single_entity(
    entity_name: str,
    description: str,
    llm_client: OpenAIModel,
    wiki_search_client: WikiSearch,
) -> tuple[str, None] | tuple[str, str]:
    """
    Process single entity

    """
    search_results = await wiki_search_client.search(entity_name)
    if not search_results:
        return entity_name, None
    examples = "\n".join(SEARCH_JUDGEMENT_PROMPT["EXAMPLES"])
    search_results.append("None of the above")

    search_results_str = "\n".join(
        [f"{i + 1}. {sr}" for i, sr in enumerate(search_results)]
    )
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

    logger.info(
        "Entity %s search result: %s response: %s",
        entity_name,
        str(search_results),
        response,
    )

    return entity_name, response


async def search_wikipedia(
    llm_client: OpenAIModel,
    wiki_search_client: WikiSearch,
    kg_instance: NetworkXStorage,
) -> dict:
    """
    Search wikipedia for entities

    :param llm_client: LLM model
    :param wiki_search_client: wiki search client
    :param kg_instance: knowledge graph instance
    :return: nodes with search results
    """
    nodes = await kg_instance.get_all_nodes()
    nodes = list(nodes)
    wiki_data = {}

    async for node in tqdm_async(nodes, desc="Searching Wikipedia", total=len(nodes)):
        entity_name = node[0].strip('"')
        description = node[1]["description"]
        try:
            entity, summary = await _process_single_entity(
                entity_name, description, llm_client, wiki_search_client
            )
            wiki_data[entity] = summary
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error processing entity %s: %s", entity_name, str(e))
    return wiki_data
