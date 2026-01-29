import json
import re
from typing import Any, Dict

from graphgen.bases import BaseLLMWrapper, BaseTripleEvaluator
from graphgen.templates import ACCURACY_EVALUATION_PROMPT
from graphgen.utils import detect_main_language, logger


class AccuracyEvaluator(BaseTripleEvaluator):
    """Evaluates accuracy of entity recognition and relation extraction using LLM-as-a-Judge.

    For each chunk, uses LLM to evaluate the quality of extracted entities and relations
    by comparing them with the original chunk content. Provides multi-dimensional quality
    scores (accuracy, completeness, precision).
    """

    def __init__(
        self,
        llm_client: BaseLLMWrapper,
    ):
        self.llm_client = llm_client

    async def evaluate(self, unit: tuple) -> Dict[str, Any]:
        """Evaluate entity and relation extraction quality using LLM-as-a-Judge.

        Returns:
            Dictionary containing entity_accuracy and relation_accuracy metrics.
        """
        chunk_content, nodes, edges = unit
        lang = detect_main_language(chunk_content)

        # node
        prompt = ACCURACY_EVALUATION_PROMPT[lang]["ENTITY"].format(
            chunk_content=chunk_content,
            extracted_entities=json.dumps(nodes, ensure_ascii=False, indent=2),
        )

        response = await self.llm_client.generate_answer(prompt)

        # Try to parse JSON response
        try:
            node_evaluation_result = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks or other formats
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                node_evaluation_result = json.loads(json_match.group(0))
            else:
                logger.warning("Failed to parse LLM response.")
                # default evaluation
                node_evaluation_result = {
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "precision": 0.0,
                    "overall_score": 0.0,
                    "accuracy_reasoning": "Failed to parse LLM response",
                    "completeness_reasoning": "",
                    "precision_reasoning": "",
                    "issues": ["LLM response parsing failed"],
                }

        # edge
        prompt = ACCURACY_EVALUATION_PROMPT[lang]["RELATION"].format(
            chunk_content=chunk_content,
            extracted_relations=json.dumps(edges, ensure_ascii=False, indent=2),
        )
        response = await self.llm_client.generate_answer(prompt)
        # Try to parse JSON response
        try:
            edge_evaluation_result = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks or other formats
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                edge_evaluation_result = json.loads(json_match.group(0))
            else:
                logger.warning("Failed to parse LLM response.")
                # default evaluation
                edge_evaluation_result = {
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "precision": 0.0,
                    "overall_score": 0.0,
                    "accuracy_reasoning": "Failed to parse LLM response",
                    "completeness_reasoning": "",
                    "precision_reasoning": "",
                    "issues": ["LLM response parsing failed"],
                }

        return {
            "entity_accuracy": node_evaluation_result,
            "relation_accuracy": edge_evaluation_result,
        }
