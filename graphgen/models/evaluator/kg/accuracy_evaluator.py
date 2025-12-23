import asyncio
from typing import Any, Dict, List, Tuple

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.models.evaluator.kg.utils import get_relevant_text, sample_items
from graphgen.utils import create_event_loop, logger


class AccuracyEvaluator:
    """Evaluates accuracy of entity recognition, relation extraction, and triple validation.
    
    Note: Recall is approximated as equal to precision since we cannot calculate true recall
    (TP / (TP + FN)) without complete ground truth. The F1 score is therefore equal to precision.
    Only precision should be considered as the primary metric.
    """

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        chunk_storage: BaseKVStorage,
        llm_client: BaseLLMWrapper,
        sample_size: int = 100,
        max_concurrent: int = 10,
    ):
        self.graph_storage = graph_storage
        self.chunk_storage = chunk_storage
        self.llm_client = llm_client
        self.sample_size = sample_size
        self.max_concurrent = max_concurrent

    def evaluate(self) -> Dict[str, Any]:
        # Get all nodes and edges
        all_nodes = self.graph_storage.get_all_nodes() or []
        all_edges = self.graph_storage.get_all_edges() or []

        if not all_nodes and not all_edges:
            return {"error": "Empty graph"}

        # Sample entities and triples (edges)
        entity_samples = sample_items(all_nodes, self.sample_size)
        triple_samples = sample_items(all_edges, self.sample_size)

        # Evaluate each type (async)
        loop = create_event_loop()
        entity_results = loop.run_until_complete(self._evaluate_entities(entity_samples))
        triple_results = loop.run_until_complete(self._evaluate_triples(triple_samples))

        return {
            "entity_accuracy": entity_results,
            "triple_accuracy": triple_results,
        }

    async def _evaluate_entities(
        self, entity_samples: List[Tuple[str, Dict]]
    ) -> Dict[str, float]:
        """Evaluate entity recognition accuracy."""
        source_text = get_relevant_text(self.chunk_storage)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def verify_with_semaphore(entity_sample):
            async with semaphore:
                entity_id, entity_data = entity_sample
                return await self._verify_entity_with_llm(
                    entity_id, entity_data, source_text
                )

        results = []
        tasks = [verify_with_semaphore(sample) for sample in entity_samples]
        for coro in tqdm_async(
            asyncio.as_completed(tasks), total=len(tasks), desc="Verifying entities"
        ):
            result = await coro
            results.append(result)

        return self._calculate_metrics(results)

    async def _evaluate_triples(
        self, triple_samples: List[Tuple[str, str, Dict]]
    ) -> Dict[str, float]:
        """Evaluate triple validation accuracy (RLC)."""
        source_text = get_relevant_text(self.chunk_storage)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def verify_with_semaphore(triple_sample):
            async with semaphore:
                src_id, dst_id, edge_data = triple_sample
                return await self._verify_triple_with_llm(
                    src_id, dst_id, edge_data, source_text
                )

        results = []
        tasks = [verify_with_semaphore(sample) for sample in triple_samples]
        for coro in tqdm_async(
            asyncio.as_completed(tasks), total=len(tasks), desc="Verifying triples"
        ):
            result = await coro
            results.append(result)

        return self._calculate_metrics(results)

    async def _verify_entity_with_llm(
        self, entity_id: str, entity_data: Dict, source_text: str
    ) -> bool:
        """Verify entity correctness using LLM."""
        entity_name = entity_data.get("entity_name", entity_id)
        entity_type = entity_data.get("entity_type", "unknown")
        entity_summary = entity_data.get("entity_summary", entity_data.get("description", ""))

        # Try to get relevant text from source_id
        source_id = entity_data.get("source_id")
        if source_id:
            relevant_text = get_relevant_text(self.chunk_storage, source_id)
            if relevant_text:
                source_text = relevant_text

        prompt = f"""给定以下文本和实体信息，请判断该实体是否在文本中正确识别。

文本：{source_text[:2000]}

实体名称：{entity_name}
实体类型：{entity_type}
实体描述：{entity_summary}

请回答：该实体是否在文本中正确识别？回答"是"或"否"，并简要说明理由。"""

        try:
            response = await self.llm_client.generate_answer(prompt)
            response_lower = response.lower()
            return (
                "是" in response_lower
                or "yes" in response_lower
                or "正确" in response_lower
            )
        except Exception as e:
            logger.error(f"LLM verification failed for entity {entity_id}: {e}")
            return False

    def _calculate_metrics(self, results: List[bool]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score from boolean verification results.
        
        Note: Recall is approximated as equal to precision since we cannot calculate
        true recall (TP / (TP + FN)) without complete ground truth. The F1 score
        is therefore equal to precision. Only precision should be considered as the
        primary metric.
        
        Args:
            results: List of boolean values indicating verification results (True = correct)
            
        Returns:
            Dictionary containing precision, recall, f1, true_positives, false_positives,
            and sample_size
        """
        tp = sum(results)
        fp = len(results) - tp
        precision = tp / len(results) if results else 0.0
        # Approximation: assume all sampled are relevant (cannot calculate true recall)
        recall = precision
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "sample_size": len(results),
        }

    async def _verify_triple_with_llm(
        self, src_id: str, dst_id: str, edge_data: Dict, source_text: str
    ) -> bool:
        """Verify triple correctness using LLM."""
        src_node = self.graph_storage.get_node(src_id) or {}
        dst_node = self.graph_storage.get_node(dst_id) or {}
        head = src_node.get("entity_name", src_id)
        tail = dst_node.get("entity_name", dst_id)
        relation = edge_data.get("relationship_summary", edge_data.get("description", ""))

        # Try to get relevant text from source_id
        source_id = edge_data.get("source_id")
        if source_id:
            relevant_text = get_relevant_text(self.chunk_storage, source_id)
            if relevant_text:
                source_text = relevant_text

        prompt = f"""给定以下文本和三元组，请判断该三元组是否正确。

文本：{source_text[:2000]}

三元组：(头实体: {head}, 关系: {relation}, 尾实体: {tail})

请回答：该三元组是否正确？回答"是"或"否"，并简要说明理由。"""

        try:
            response = await self.llm_client.generate_answer(prompt)
            response_lower = response.lower()
            return (
                "是" in response_lower
                or "yes" in response_lower
                or "正确" in response_lower
            )
        except Exception as e:
            logger.error(f"LLM verification failed for triple {src_id}->{dst_id}: {e}")
            return False
