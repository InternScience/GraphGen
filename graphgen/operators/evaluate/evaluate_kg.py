from typing import Any, Dict

from graphgen.bases import BaseGraphStorage
from graphgen.utils import logger


def evaluate_kg(
    kg_evaluators: Dict[str, Any],
    kg_instance: BaseGraphStorage,
) -> Dict[str, Any]:
    results = {}
    for key, kg_evaluator in kg_evaluators.items():
        results[key] = kg_evaluator.evaluate(kg_instance)
        logger.info(f"KG Evaluation result for {key}: {results[key]}")
    return results
