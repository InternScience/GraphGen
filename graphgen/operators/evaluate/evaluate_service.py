from typing import Tuple

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger

from .evaluate_kg import evaluate_kg
from .evaluate_qa import evaluate_qa
from .evaluate_triple import evaluate_triple


class EvaluateService(BaseOperator):
    """
    1. KG Quality Evaluation
    2. QA Quality Evaluation
    """

    def __init__(
        self,
        target: str,
        metrics: list[str],
        working_dir: str = "cache",
        graph_backend: str = "kuzu",
        kv_backend: str = "rocksdb",
        **kwargs,
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="evaluate"
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.metrics = metrics or []
        self.kwargs = kwargs
        self.graph_storage = init_storage(
            backend=graph_backend, working_dir=working_dir, namespace="graph"
        )

        # Initialize evaluators
        self.target = target
        self.src_storage = None
        self.tgt_storage = None
        self.evaluators = {}
        self._init_evaluators(self.target, metrics)

    def _init_evaluators(self, target: str, metrics: list[str]):
        """Initialize evaluators based on target and metrics."""
        if target not in {"qa", "kg", "triple"}:
            raise ValueError(f"Unknown evaluation target: {target}")

        # Delegate to target-specific initializer
        getattr(self, f"_init_{target}_evaluators")(metrics)

    def _init_qa_evaluators(self, metrics: list[str]):
        """Initialize QA evaluators."""
        for metric in metrics:
            self.evaluators[metric] = self._create_qa_evaluator(metric)

    def _create_qa_evaluator(self, metric: str):
        """Factory method for QA evaluator instances."""
        if metric == "length":
            from graphgen.models import LengthEvaluator

            return LengthEvaluator()
        if metric == "mtld":
            from graphgen.models import MTLDEvaluator

            return MTLDEvaluator(**self.kwargs.get("mtld_params", {}))
        if metric == "reward_score":
            from graphgen.models import RewardEvaluator

            return RewardEvaluator(**self.kwargs.get("reward_params", {}))
        if metric == "uni_score":
            from graphgen.models import UniEvaluator

            return UniEvaluator(**self.kwargs.get("uni_params", {}))
        raise ValueError(f"Unknown QA metric: {metric}")

    def _init_kg_evaluators(self, metrics: list[str]):
        """Initialize KG evaluators."""
        for metric in metrics:
            if metric != "structure":
                raise ValueError(f"Unknown KG metric: {metric}")
            from graphgen.models import StructureEvaluator

            self.evaluators[metric] = StructureEvaluator(
                **self.kwargs.get("structure_params", {})
            )

    def _init_triple_evaluators(self, metrics: list[str]):
        """Initialize Triple evaluators."""
        self.src_storage = init_storage(
            backend=self.kv_backend,
            working_dir=self.working_dir,
            namespace=self.kwargs["src_namespace"],
        )
        self.tgt_storage = init_storage(
            backend=self.kv_backend,
            working_dir=self.working_dir,
            namespace=self.kwargs["tgt_namespace"],
        )

        for metric in metrics:
            if metric != "accuracy":
                raise ValueError(f"Unknown Triple metric: {metric}")
            from graphgen.models import AccuracyEvaluator

            self.evaluators[metric] = AccuracyEvaluator(llm_client=self.llm_client)

    def process(self, batch: list) -> Tuple[list, dict]:
        final_results = []
        meta_updates = {}

        # 1. QA Evaluation (per item)
        if self.target == "qa" and self.evaluators:
            results: dict = evaluate_qa(self.evaluators, batch)
            for i, item in enumerate(batch):
                metrics = {}
                for _, scores in results.items():
                    metrics.update(scores[i])
                item.update({"metrics": metrics})
                input_trace_id = item.pop("_trace_id")
                item["_trace_id"] = self.get_trace_id(item)
                final_results.append(item)
                meta_updates.setdefault(input_trace_id, []).append(item["_trace_id"])

            return final_results, meta_updates

        # 2. KG evaluation
        if self.target == "kg" and self.evaluators:
            results = evaluate_kg(
                self.evaluators,
                self.graph_storage,
            )
            if not results:
                logger.warning("No KG evaluation results, returning empty DataFrame")
                return [], {}
            results["_trace_id"] = self.get_trace_id(results)
            final_results.append(results)
            return final_results, {}

        # 3. Triple evaluation
        if self.target == "triple" and self.evaluators:
            results = evaluate_triple(
                self.evaluators, self.src_storage, self.tgt_storage
            )
            results["_trace_id"] = "evaluate-triple-result"
            final_results.append(results)
            return final_results, {}

        # No metrics specified
        logger.warning("No metrics specified, returning empty DataFrame")
        return [], {}
