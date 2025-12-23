import asyncio
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.models.evaluator.kg.utils import load_text_content
from graphgen.utils import create_event_loop, logger


class AccuracyEvaluator:
    """Evaluates accuracy of entity recognition, relation extraction, and triple validation.
    
    Uses LLM to extract ground truth (entities, relations, and triples) from source texts,
    then compares with entities, relations, and triples in the knowledge graph to calculate
    true precision, recall, and F1 scores.
    
    Three evaluation dimensions:
    1. Entity recognition accuracy: compares extracted entities with KG entities
    2. Relation extraction accuracy: compares extracted relation descriptions with KG relation descriptions
    3. Triple validation (RLC): validates complete triples (head entity + relation + tail entity)
    """
    
    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        chunk_storage: BaseKVStorage,
        llm_client: BaseLLMWrapper,
        source_text_paths: List[str],
        max_concurrent: int = 10,
    ):
        if not source_text_paths:
            raise ValueError("source_text_paths is required and cannot be empty")
        self.graph_storage = graph_storage
        self.chunk_storage = chunk_storage
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent
        self.source_text_paths = source_text_paths

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate entity, relation, and triple accuracy.
        
        Returns:
            Dictionary containing entity_accuracy, relation_accuracy, and triple_accuracy metrics.
        """
        all_nodes = self.graph_storage.get_all_nodes() or []
        all_edges = self.graph_storage.get_all_edges() or []

        if not all_nodes and not all_edges:
            return {"error": "Empty graph"}

        loop = create_event_loop()
        source_text = self._load_source_texts()
        entity_ground_truth, relation_ground_truth, triple_ground_truth = loop.run_until_complete(
            self._extract_ground_truth(source_text)
        )
        entity_results = loop.run_until_complete(
            self._evaluate_entities_with_ground_truth(
                all_nodes, entity_ground_truth
            )
        )
        relation_results = loop.run_until_complete(
            self._evaluate_relations_with_ground_truth(
                all_edges, relation_ground_truth
            )
        )
        triple_results = loop.run_until_complete(
            self._evaluate_triples_with_ground_truth(
                all_edges, triple_ground_truth
            )
        )

        return {
            "entity_accuracy": entity_results,
            "relation_accuracy": relation_results,
            "triple_accuracy": triple_results,
        }

    def _load_source_texts(self) -> str:
        """Load and concatenate source text files.
        
        Supports .txt, .json, and .jsonl formats using the utility function
        from graphgen.models.evaluator.kg.utils.
        """
        texts = []
        for path in self.source_text_paths:
            file_path = Path(path)
            if not file_path.exists():
                logger.warning(f"Source text file not found: {path}")
                continue
            try:
                content = load_text_content(file_path)
                if content:
                    texts.append(content)
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
        return "\n\n".join(texts)

    async def _extract_ground_truth(
        self, source_text: str
    ) -> Tuple[Set[str], Set[str], Set[Tuple[str, str, str]]]:
        """Extract entities, relations, and triples from source text using LLM as ground truth."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        chunk_size = 2000
        chunks = [
            source_text[i : i + chunk_size]
            for i in range(0, len(source_text), chunk_size)
        ]

        async def extract_from_chunk(chunk):
            async with semaphore:
                return await self._extract_entities_relations_and_triples(chunk)

        tasks = [extract_from_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        all_entities = set()
        all_relations = set()
        all_triples = set()
        for entities, relations, triples in results:
            all_entities.update(entities)
            all_relations.update(relations)
            all_triples.update(triples)

        return all_entities, all_relations, all_triples

    async def _extract_entities_relations_and_triples(
        self, text: str
    ) -> Tuple[Set[str], Set[str], Set[Tuple[str, str, str]]]:
        """Extract entities, relations, and triples from a text chunk using LLM."""
        entity_prompt = f"""从以下文本中提取所有实体名称，每行一个实体名称。

文本：
{text[:2000]}

请只返回实体名称列表，每行一个，不要其他内容："""

        relation_prompt = f"""从以下文本中提取所有关系描述，每行一个关系描述。
关系描述是指描述两个实体之间关系的词语或短语，例如"设计"、"位于"、"属于"等。

文本：
{text[:2000]}

请只返回关系描述列表，每行一个，不要其他内容："""

        triple_prompt = f"""从以下文本中提取所有三元组，格式为：头实体|关系|尾实体，每行一个三元组。

文本：
{text[:2000]}

请只返回三元组列表，每行一个，格式为：头实体|关系|尾实体，不要其他内容："""

        try:
            entity_response = await self.llm_client.generate_answer(entity_prompt)
            relation_response = await self.llm_client.generate_answer(relation_prompt)
            triple_response = await self.llm_client.generate_answer(triple_prompt)

            entities = {
                line.strip()
                for line in entity_response.split("\n")
                if line.strip() and not line.strip().startswith("#")
            }

            relations = {
                line.strip()
                for line in relation_response.split("\n")
                if line.strip() and not line.strip().startswith("#")
            }

            triples = set()
            for line in triple_response.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) == 3:
                    triples.add((parts[0].strip(), parts[1].strip(), parts[2].strip()))

            return entities, relations, triples
        except Exception as e:
            logger.error(f"Failed to extract ground truth: {e}")
            return set(), set(), set()

    async def _evaluate_entities_with_ground_truth(
        self, all_nodes: List[Tuple[str, Dict]], ground_truth: Set[str]
    ) -> Dict[str, float]:
        """Evaluate entity accuracy by comparing KG entities with ground truth."""
        kg_entities = {
            node_data.get("entity_name", node_id).lower()
            for node_id, node_data in all_nodes
            if isinstance(node_data, dict)
        }

        tp = len(kg_entities & {e.lower() for e in ground_truth})
        fp = len(kg_entities - {e.lower() for e in ground_truth})
        fn = len({e.lower() for e in ground_truth} - kg_entities)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_ground_truth": len(ground_truth),
            "total_kg_entities": len(kg_entities),
        }

    async def _evaluate_relations_with_ground_truth(
        self, all_edges: List[Tuple[str, str, Dict]], ground_truth: Set[str]
    ) -> Dict[str, float]:
        """Evaluate relation extraction accuracy by comparing KG relation descriptions with ground truth."""
        kg_relations = set()
        for src_id, dst_id, edge_data in all_edges:
            relation = edge_data.get("relationship_summary", edge_data.get("description", ""))
            if relation:
                kg_relations.add(relation.lower().strip())

        gt_normalized = {r.lower().strip() for r in ground_truth if r.strip()}

        tp = len(kg_relations & gt_normalized)
        fp = len(kg_relations - gt_normalized)
        fn = len(gt_normalized - kg_relations)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_ground_truth": len(ground_truth),
            "total_kg_relations": len(kg_relations),
        }

    async def _evaluate_triples_with_ground_truth(
        self, all_edges: List[Tuple[str, str, Dict]], ground_truth: Set[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        """Evaluate triple accuracy by comparing KG triples with ground truth."""
        kg_triples = set()
        for src_id, dst_id, edge_data in all_edges:
            src_node = self.graph_storage.get_node(src_id) or {}
            dst_node = self.graph_storage.get_node(dst_id) or {}
            head = src_node.get("entity_name", src_id).lower()
            tail = dst_node.get("entity_name", dst_id).lower()
            relation = edge_data.get("relationship_summary", edge_data.get("description", "")).lower()
            kg_triples.add((head, relation, tail))

        gt_normalized = {(h.lower(), r.lower(), t.lower()) for h, r, t in ground_truth}

        tp = len(kg_triples & gt_normalized)
        fp = len(kg_triples - gt_normalized)
        fn = len(gt_normalized - kg_triples)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_ground_truth": len(ground_truth),
            "total_kg_triples": len(kg_triples),
        }

