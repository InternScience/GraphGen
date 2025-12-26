import asyncio
import json
import re
from typing import Any, Dict, List

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.templates import ACCURACY_EVALUATION_PROMPT
from graphgen.utils import detect_main_language, logger


class AccuracyEvaluator:
    """Evaluates accuracy of entity recognition and relation extraction using LLM-as-a-Judge.

    For each chunk, uses LLM to evaluate the quality of extracted entities and relations
    by comparing them with the original chunk content. Provides multi-dimensional quality
    scores (accuracy, completeness, precision).
    """

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        chunk_storage: BaseKVStorage,
        llm_client: BaseLLMWrapper,
    ):
        self.graph_storage = graph_storage
        self.chunk_storage = chunk_storage
        self.llm_client = llm_client

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate entity and relation extraction quality using LLM-as-a-Judge.

        Returns:
            Dictionary containing entity_accuracy and relation_accuracy metrics.
        """
        # 1. Load all chunks from storage
        chunks = self._load_chunks_from_storage()

        if not chunks:
            logger.warning("No chunks found in storage")
            return {"error": "No chunks found in storage"}

        logger.info(f"Found {len(chunks)} chunks to evaluate")

        # 2. Evaluate each chunk
        entity_evaluations, relation_evaluations = self._evaluate_all_chunks(chunks)

        # 3. Aggregate results
        return self._aggregate_evaluation_results(
            entity_evaluations, relation_evaluations
        )

    def _load_chunks_from_storage(self) -> List[Chunk]:
        """Load all chunks from chunk storage."""
        chunks = []
        all_chunk_data = self.chunk_storage.get_all()

        for chunk_id, chunk_data in all_chunk_data.items():
            try:
                chunk = Chunk.from_dict(chunk_id, chunk_data)
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to load chunk {chunk_id}: {e}")
                continue

        return chunks

    def _get_extracted_entities_for_chunk(self, chunk_id: str) -> List[Dict]:
        """Get all entities extracted from the specified chunk."""
        entities = []
        all_nodes = self.graph_storage.get_all_nodes() or []

        for node_id, node_data in all_nodes:
            if not isinstance(node_data, dict):
                continue
            source_ids = node_data.get("source_id", "").split("<SEP>")
            # Check if this chunk_id is in the source_ids
            if chunk_id in [sid.strip() for sid in source_ids if sid.strip()]:
                entities.append(
                    {
                        "entity_name": node_data.get("entity_name", node_id),
                        "entity_type": node_data.get("entity_type", ""),
                        "description": node_data.get("description", ""),
                    }
                )

        return entities

    def _get_extracted_relations_for_chunk(self, chunk_id: str) -> List[Dict]:
        """Get all relations extracted from the specified chunk."""
        relations = []
        all_edges = self.graph_storage.get_all_edges() or []

        for src_id, dst_id, edge_data in all_edges:
            if not isinstance(edge_data, dict):
                continue
            source_ids = edge_data.get("source_id", "").split("<SEP>")
            # Check if this chunk_id is in the source_ids
            if chunk_id in [sid.strip() for sid in source_ids if sid.strip()]:
                src_node = self.graph_storage.get_node(src_id) or {}
                dst_node = self.graph_storage.get_node(dst_id) or {}
                relations.append(
                    {
                        "source_entity": src_node.get("entity_name", src_id),
                        "target_entity": dst_node.get("entity_name", dst_id),
                        "relationship_summary": edge_data.get("description", ""),
                    }
                )

        return relations

    def _evaluate_all_chunks(
        self, chunks: List[Chunk]
    ) -> tuple[List[Dict], List[Dict]]:
        """Evaluate all chunks sequentially."""
        entity_evaluations = []
        relation_evaluations = []

        for chunk in chunks:
            try:
                entities = self._get_extracted_entities_for_chunk(chunk.id)
                relations = self._get_extracted_relations_for_chunk(chunk.id)

                entity_eval = self._evaluate_entity_extraction(chunk, entities)
                relation_eval = self._evaluate_relation_extraction(chunk, relations)

                entity_evaluations.append(entity_eval)
                relation_evaluations.append(relation_eval)
            except Exception as e:
                logger.error(f"Failed to evaluate chunk {chunk.id}: {e}")
                continue

        return entity_evaluations, relation_evaluations

    def _evaluate_entity_extraction(
        self, chunk: Chunk, extracted_entities: List[Dict]
    ) -> Dict[str, Any]:
        """Use LLM to evaluate entity extraction quality."""
        try:
            lang = detect_main_language(chunk.content)

            prompt = ACCURACY_EVALUATION_PROMPT[lang]["ENTITY"].format(
                chunk_content=chunk.content,
                extracted_entities=json.dumps(
                    extracted_entities, ensure_ascii=False, indent=2
                ),
            )

            response = asyncio.run(self.llm_client.generate_answer(prompt))

            # Try to parse JSON response
            try:
                evaluation_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks or other formats
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    evaluation_result = json.loads(json_match.group(0))
                else:
                    logger.warning(
                        f"Failed to parse LLM response for chunk {chunk.id}: {response[:200]}"
                    )
                    # Return default evaluation
                    evaluation_result = {
                        "accuracy": 0.0,
                        "completeness": 0.0,
                        "precision": 0.0,
                        "overall_score": 0.0,
                        "accuracy_reasoning": "Failed to parse LLM response",
                        "completeness_reasoning": "",
                        "precision_reasoning": "",
                        "issues": ["LLM response parsing failed"],
                    }

            # Validate and calculate overall_score if not provided
            if "overall_score" not in evaluation_result:
                accuracy = float(evaluation_result.get("accuracy", 0.0))
                completeness = float(evaluation_result.get("completeness", 0.0))
                precision = float(evaluation_result.get("precision", 0.0))
                evaluation_result["overall_score"] = (
                    0.4 * accuracy + 0.4 * completeness + 0.2 * precision
                )

            return {
                "chunk_id": chunk.id,
                "chunk_content": chunk.content[:200]
                if chunk.content
                else "",  # First 200 chars for debugging
                "extracted_entities_count": len(extracted_entities),
                **evaluation_result,
            }
        except Exception as e:
            logger.error(
                f"Error evaluating entity extraction for chunk {chunk.id}: {e}"
            )
            return {
                "chunk_id": chunk.id,
                "chunk_content": chunk.content[:200] if chunk.content else "",
                "extracted_entities_count": len(extracted_entities),
                "accuracy": 0.0,
                "completeness": 0.0,
                "precision": 0.0,
                "overall_score": 0.0,
                "accuracy_reasoning": f"Evaluation failed: {str(e)}",
                "completeness_reasoning": "",
                "precision_reasoning": "",
                "issues": [f"Evaluation error: {str(e)}"],
            }

    def _evaluate_relation_extraction(
        self, chunk: Chunk, extracted_relations: List[Dict]
    ) -> Dict[str, Any]:
        """Use LLM to evaluate relation extraction quality."""
        try:
            lang = detect_main_language(chunk.content)
            prompt = ACCURACY_EVALUATION_PROMPT[lang]["RELATION"].format(
                chunk_content=chunk.content,
                extracted_relations=json.dumps(
                    extracted_relations, ensure_ascii=False, indent=2
                ),
            )

            response = asyncio.run(self.llm_client.generate_answer(prompt))

            # Try to parse JSON response
            try:
                evaluation_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks or other formats
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    evaluation_result = json.loads(json_match.group(0))
                else:
                    logger.warning(
                        f"Failed to parse LLM response for chunk {chunk.id}: {response[:200]}"
                    )
                    # Return default evaluation
                    evaluation_result = {
                        "accuracy": 0.0,
                        "completeness": 0.0,
                        "precision": 0.0,
                        "overall_score": 0.0,
                        "accuracy_reasoning": "Failed to parse LLM response",
                        "completeness_reasoning": "",
                        "precision_reasoning": "",
                        "issues": ["LLM response parsing failed"],
                    }

            # Validate and calculate overall_score if not provided
            if "overall_score" not in evaluation_result:
                accuracy = float(evaluation_result.get("accuracy", 0.0))
                completeness = float(evaluation_result.get("completeness", 0.0))
                precision = float(evaluation_result.get("precision", 0.0))
                evaluation_result["overall_score"] = (
                    0.4 * accuracy + 0.4 * completeness + 0.2 * precision
                )

            return {
                "chunk_id": chunk.id,
                "chunk_content": chunk.content[:200] if chunk.content else "",
                "extracted_relations_count": len(extracted_relations),
                **evaluation_result,
            }
        except Exception as e:
            logger.error(
                f"Error evaluating relation extraction for chunk {chunk.id}: {e}"
            )
            return {
                "chunk_id": chunk.id,
                "chunk_content": chunk.content[:200] if chunk.content else "",
                "extracted_relations_count": len(extracted_relations),
                "accuracy": 0.0,
                "completeness": 0.0,
                "precision": 0.0,
                "overall_score": 0.0,
                "accuracy_reasoning": f"Evaluation failed: {str(e)}",
                "completeness_reasoning": "",
                "precision_reasoning": "",
                "issues": [f"Evaluation error: {str(e)}"],
            }

    @staticmethod
    def _aggregate_evaluation_results(
        entity_evaluations: List[Dict], relation_evaluations: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate evaluation results from all chunks."""

        def calculate_stats(scores: List[float]) -> Dict[str, float]:
            if not scores:
                return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            sorted_scores = sorted(scores)
            n = len(scores)
            mean = sum(scores) / n
            median = (
                sorted_scores[n // 2]
                if n % 2 == 1
                else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            )
            variance = sum((x - mean) ** 2 for x in scores) / n
            std = variance**0.5

            return {
                "mean": mean,
                "median": median,
                "min": min(scores),
                "max": max(scores),
                "std": std,
            }

        # Extract scores
        entity_overall_scores = [
            e.get("overall_score", 0.0) for e in entity_evaluations
        ]
        entity_accuracy_scores = [e.get("accuracy", 0.0) for e in entity_evaluations]
        entity_completeness_scores = [
            e.get("completeness", 0.0) for e in entity_evaluations
        ]
        entity_precision_scores = [e.get("precision", 0.0) for e in entity_evaluations]

        relation_overall_scores = [
            r.get("overall_score", 0.0) for r in relation_evaluations
        ]
        relation_accuracy_scores = [
            r.get("accuracy", 0.0) for r in relation_evaluations
        ]
        relation_completeness_scores = [
            r.get("completeness", 0.0) for r in relation_evaluations
        ]
        relation_precision_scores = [
            r.get("precision", 0.0) for r in relation_evaluations
        ]

        return {
            "entity_accuracy": {
                "overall_score": calculate_stats(entity_overall_scores),
                "accuracy": calculate_stats(entity_accuracy_scores),
                "completeness": calculate_stats(entity_completeness_scores),
                "precision": calculate_stats(entity_precision_scores),
                "total_chunks": len(entity_evaluations),
                "detailed_results": entity_evaluations,
            },
            "relation_accuracy": {
                "overall_score": calculate_stats(relation_overall_scores),
                "accuracy": calculate_stats(relation_accuracy_scores),
                "completeness": calculate_stats(relation_completeness_scores),
                "precision": calculate_stats(relation_precision_scores),
                "total_chunks": len(relation_evaluations),
                "detailed_results": relation_evaluations,
            },
        }
