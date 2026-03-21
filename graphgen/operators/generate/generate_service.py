import copy
import json
from typing import Tuple

from graphgen.bases import BaseKVStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger, run_concurrent


class GenerateService(BaseOperator):
    """
    Generate question-answer pairs based on nodes and edges.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        method: str = "aggregated",
        data_format: str = "ChatML",
        **generate_kwargs,
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="generate"
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.generate_storage: BaseKVStorage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="generate"
        )

        self.method = method
        self.data_format = data_format

        if self.method == "atomic":
            from graphgen.models import AtomicGenerator

            self.generator = AtomicGenerator(self.llm_client)
        elif self.method == "aggregated":
            from graphgen.models import AggregatedGenerator

            self.generator = AggregatedGenerator(self.llm_client)
        elif self.method == "multi_hop":
            from graphgen.models import MultiHopGenerator

            self.generator = MultiHopGenerator(self.llm_client)
        elif self.method == "cot":
            from graphgen.models import CoTGenerator

            self.generator = CoTGenerator(self.llm_client)
        elif self.method == "vqa":
            from graphgen.models import VQAGenerator

            self.generator = VQAGenerator(self.llm_client)
        elif self.method == "multi_choice":
            from graphgen.models import MultiChoiceGenerator

            self.generator = MultiChoiceGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        elif self.method == "multi_answer":
            from graphgen.models import MultiAnswerGenerator

            self.generator = MultiAnswerGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 3),
            )
        elif self.method == "fill_in_blank":
            from graphgen.models import FillInBlankGenerator

            self.generator = FillInBlankGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        elif self.method == "true_false":
            from graphgen.models import TrueFalseGenerator

            self.generator = TrueFalseGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        else:
            raise ValueError(f"Unsupported generation mode: {method}")

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Generate question-answer pairs based on nodes and edges.
        """
        logger.info("[Generation] mode: %s, batches: %d", self.method, len(batch))
        triples = [(item["nodes"], item["edges"]) for item in batch]
        results = run_concurrent(
            self.generator.generate,
            triples,
            desc="Generating QAs",
            unit="batch",
        )

        meta_updates = {}
        final_results = []
        for item, qa_pairs in zip(batch, results):
            if not qa_pairs:
                continue
            input_trace_id = item["_trace_id"]
            sub_graph = {
                "nodes": copy.deepcopy(item.get("nodes", [])),
                "edges": copy.deepcopy(item.get("edges", [])),
            }
            sub_graph_summary = self._build_sub_graph_summary(
                item.get("nodes", []), item.get("edges", [])
            )
            for qa_pair in qa_pairs:
                res = self.generator.format_generation_results(
                    qa_pair, output_data_format=self.data_format
                )
                res["sub_graph"] = json.dumps(sub_graph, ensure_ascii=False)
                res["sub_graph_summary"] = json.dumps(
                    sub_graph_summary, ensure_ascii=False
                )
                res["_trace_id"] = self.get_trace_id(res)
                final_results.append(res)
                meta_updates.setdefault(input_trace_id, []).append(res["_trace_id"])
        return final_results, meta_updates

    def split(self, batch):
        to_process, recovered = super().split(batch)
        if not recovered.empty and "sub_graph" in recovered.columns:
            recovered = recovered.copy()
            recovered["sub_graph"] = recovered["sub_graph"].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
        if not recovered.empty and "sub_graph_summary" in recovered.columns:
            recovered = recovered.copy()
            recovered["sub_graph_summary"] = recovered["sub_graph_summary"].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
        if not recovered.empty and "sub_graph" in recovered.columns:
            recovered = recovered.copy()
            recovered["sub_graph_summary"] = recovered.apply(
                lambda row: row.get("sub_graph_summary")
                if row.get("sub_graph_summary")
                else self._build_summary_from_serialized_sub_graph(row.get("sub_graph")),
                axis=1,
            )
        return to_process, recovered

    @staticmethod
    def _build_sub_graph_summary(nodes: list, edges: list) -> dict:
        def _node_label(node) -> str:
            if not isinstance(node, (list, tuple)) or not node:
                return str(node)
            return str(node[0])

        def _edge_label(edge) -> str:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                return str(edge)
            return f"{edge[0]} -> {edge[1]}"

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_ids": [_node_label(node) for node in nodes[:10]],
            "edge_pairs": [_edge_label(edge) for edge in edges[:10]],
        }

    @classmethod
    def _build_summary_from_serialized_sub_graph(cls, sub_graph) -> str | None:
        if not sub_graph:
            return None
        try:
            parsed = json.loads(sub_graph) if isinstance(sub_graph, str) else sub_graph
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None
        summary = cls._build_sub_graph_summary(
            parsed.get("nodes", []), parsed.get("edges", [])
        )
        return json.dumps(summary, ensure_ascii=False)
