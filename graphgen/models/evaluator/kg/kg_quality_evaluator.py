from typing import Any, Dict

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.common import init_llm, init_storage
from graphgen.models.evaluator.kg.accuracy_evaluator import AccuracyEvaluator
from graphgen.models.evaluator.kg.consistency_evaluator import ConsistencyEvaluator
from graphgen.models.evaluator.kg.structure_evaluator import StructureEvaluator
from graphgen.utils import logger


class KGQualityEvaluator:
    def __init__(
        self,
        working_dir: str = "cache",
        graph_backend: str = "kuzu",
        kv_backend: str = "rocksdb",
        **kwargs
    ):
        # Initialize storage
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )
        self.chunk_storage: BaseKVStorage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="chunk"
        )
        
        # Initialize LLM client
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        
        # Initialize individual evaluators
        self.accuracy_evaluator = AccuracyEvaluator(
            graph_storage=self.graph_storage,
            chunk_storage=self.chunk_storage,
            llm_client=self.llm_client,
        )
        
        self.consistency_evaluator = ConsistencyEvaluator(
            graph_storage=self.graph_storage,
            chunk_storage=self.chunk_storage,
            llm_client=self.llm_client,
        )
        
        # Structure evaluator doesn't need chunk_storage or llm_client
        structure_params = kwargs.get("structure_params", {})
        self.structure_evaluator = StructureEvaluator(
            graph_storage=self.graph_storage,
            **structure_params
        )
        
        logger.info("KGQualityEvaluator initialized")

    def evaluate_accuracy(self) -> Dict[str, Any]:
        logger.info("Running accuracy evaluation...")
        results = self.accuracy_evaluator.evaluate()
        logger.info("Accuracy evaluation completed")
        return results

    def evaluate_consistency(self) -> Dict[str, Any]:
        logger.info("Running consistency evaluation...")
        results = self.consistency_evaluator.evaluate()
        logger.info("Consistency evaluation completed")
        return results

    def evaluate_structure(self) -> Dict[str, Any]:
        logger.info("Running structural robustness evaluation...")
        results = self.structure_evaluator.evaluate()
        logger.info("Structural robustness evaluation completed")
        return results

    def evaluate_all(self) -> Dict[str, Any]:
        logger.info("Running all KG evaluations...")
        results = {
            "accuracy": self.evaluate_accuracy(),
            "consistency": self.evaluate_consistency(),
            "structure": self.evaluate_structure(),
        }
        logger.info("All KG evaluations completed")
        return results

