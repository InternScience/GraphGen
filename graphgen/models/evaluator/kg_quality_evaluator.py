"""
Knowledge Graph Quality Evaluator

This module provides comprehensive quality evaluation for knowledge graphs,
1. accuracy assessment (entity/relation/triple validation),
2. consistency assessment (attribute conflict detection), and structural
3. robustness assessment (noise ratio, connectivity, degree distribution).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.models.evaluator.kg import (
    AccuracyEvaluator,
    ConsistencyEvaluator,
    StructureEvaluator,
)
from graphgen.utils import CURRENT_LOGGER_VAR, logger


@dataclass
class KGQualityEvaluator:
    """Knowledge Graph Quality Evaluator."""

    working_dir: str = "cache"
    graph_backend: str = "kuzu"
    kv_backend: str = "rocksdb"
    llm_client: Optional[BaseLLMWrapper] = None
    max_concurrent: int = 10

    def __post_init__(self):
        """Initialize storage and LLM client."""
        # Lazy import to avoid circular dependency
        from graphgen.common import init_llm, init_storage

        self.graph_storage: BaseGraphStorage = init_storage(
            backend=self.graph_backend,
            working_dir=self.working_dir,
            namespace="graph",
        )
        self.chunk_storage: BaseKVStorage = init_storage(
            backend=self.kv_backend,
            working_dir=self.working_dir,
            namespace="chunk",
        )

        if self.llm_client is None:
            self.llm_client = init_llm("trainee")

    def evaluate_all(self) -> Dict[str, Any]:
        """Run all evaluation metrics and return comprehensive report."""
        CURRENT_LOGGER_VAR.get()
        results = {}

        try:
            logger.info("Starting accuracy evaluation...")
            results["accuracy"] = self.evaluate_accuracy()
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            results["accuracy"] = {"error": str(e)}

        # Consistency evaluation
        try:
            logger.info("Starting consistency evaluation...")
            consistency_evaluator = ConsistencyEvaluator(
                graph_storage=self.graph_storage,
                chunk_storage=self.chunk_storage,
                llm_client=self.llm_client,
                max_concurrent=self.max_concurrent,
            )
            results["consistency"] = consistency_evaluator.evaluate()
        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            results["consistency"] = {"error": str(e)}

        try:
            logger.info("Starting structural robustness evaluation...")
            structure_evaluator = StructureEvaluator(
                graph_storage=self.graph_storage
            )
            results["structure"] = structure_evaluator.evaluate()
        except Exception as e:
            logger.error(f"Structural evaluation failed: {e}")
            results["structure"] = {"error": str(e)}

        return results

    def evaluate_accuracy(self) -> Dict[str, Any]:
        accuracy_evaluator = AccuracyEvaluator(
            graph_storage=self.graph_storage,
            chunk_storage=self.chunk_storage,
            llm_client=self.llm_client,
            max_concurrent=self.max_concurrent,
        )
        return accuracy_evaluator.evaluate()

    def evaluate_consistency(self) -> Dict[str, Any]:
        consistency_evaluator = ConsistencyEvaluator(
            graph_storage=self.graph_storage,
            chunk_storage=self.chunk_storage,
            llm_client=self.llm_client,
            max_concurrent=self.max_concurrent,
        )
        return consistency_evaluator.evaluate()

    def evaluate_structure(self) -> Dict[str, Any]:
        structure_evaluator = StructureEvaluator(
            graph_storage=self.graph_storage
        )
        return structure_evaluator.evaluate()
