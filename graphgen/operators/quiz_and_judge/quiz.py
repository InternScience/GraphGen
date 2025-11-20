import asyncio
from collections import defaultdict

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.bases import BaseLLMWrapper
from graphgen.models import JsonKVStorage, NetworkXStorage, QuizGenerator
from graphgen.utils import logger


async def quiz(
    synth_llm_client: BaseLLMWrapper,
    graph_storage: NetworkXStorage,
    rephrase_storage: JsonKVStorage,
    max_samples: int = 1,
    max_concurrent: int = 1000,
) -> JsonKVStorage:
    """
    Get all edges and quiz them using QuizGenerator.

    :param synth_llm_client: generate statements
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param max_samples: max samples for each edge
    :param max_concurrent: max concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)
    generator = QuizGenerator(synth_llm_client)

    async def _process_single_quiz(description: str, template_type: str, gt: str):
        async with semaphore:
            try:
                # if rephrase_storage exists already, directly get it
                descriptions = await rephrase_storage.get_by_id(description)
                if descriptions:
                    return None

                prompt = generator.build_prompt_for_description(description, template_type)
                new_description = await synth_llm_client.generate_answer(
                    prompt, temperature=1
                )
                rephrased_text = generator.parse_rephrased_text(new_description)
                return {description: [(rephrased_text, gt)]}

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error when quizzing description %s: %s", description, e)
                return None

    edges = await graph_storage.get_all_edges()
    nodes = await graph_storage.get_all_nodes()

    results = defaultdict(list)
    tasks = []
    for edge in edges:
        edge_data = edge[2]
        description = edge_data["description"]

        results[description] = [(description, "yes")]

        for i in range(max_samples):
            if i > 0:
                tasks.append(
                    _process_single_quiz(description, "TEMPLATE", "yes")
                )
            tasks.append(
                _process_single_quiz(description, "ANTI_TEMPLATE", "no")
            )

    for node in nodes:
        node_data = node[1]
        description = node_data["description"]

        results[description] = [(description, "yes")]

        for i in range(max_samples):
            if i > 0:
                tasks.append(
                    _process_single_quiz(description, "TEMPLATE", "yes")
                )
            tasks.append(
                _process_single_quiz(description, "ANTI_TEMPLATE", "no")
            )

    for result in tqdm_async(
        asyncio.as_completed(tasks), total=len(tasks), desc="Quizzing descriptions"
    ):
        new_result = await result
        if new_result:
            for key, value in new_result.items():
                results[key].extend(value)

    for key, value in results.items():
        results[key] = list(set(value))
        await rephrase_storage.upsert({key: results[key]})

    return rephrase_storage
