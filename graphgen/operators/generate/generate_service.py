from typing import Tuple

from graphgen.bases import BaseKVStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger, run_concurrent


class GenerateService(BaseOperator):
    """
    Generate question-answer pairs based on nodes and edges.

    PMS fork additions (all optional, default = official behavior):
    - ``prompt_profile``: versioned prompt override loaded via
      graphgen.templates.prompt_profiles;
    - ``output_gate``: deterministic QA filter registered in
      graphgen.operators.generate.gates;
    - ``llm_concurrency``: cap on in-flight LLM requests per event loop.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        method: str = "aggregated",
        data_format: str = "ChatML",
        prompt_profile: str | None = None,
        output_gate: str | None = None,
        llm_concurrency: int | None = None,
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
        elif self.method == "masked_fill_in_blank":
            from graphgen.models import MaskedFillInBlankGenerator

            self.generator = MaskedFillInBlankGenerator(self.llm_client)
        elif self.method == "true_false":
            from graphgen.models import TrueFalseGenerator

            self.generator = TrueFalseGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        else:
            raise ValueError(f"Unsupported generation mode: {method}")

        if prompt_profile:
            self.generator.apply_prompt_profile(prompt_profile)
        if llm_concurrency is not None:
            from graphgen.operators.generate.llm_concurrency import (
                bound_llm_concurrency,
            )

            limited = bound_llm_concurrency(self.generator.generate, llm_concurrency)
            self.generator.generate = limited
        if output_gate:
            from graphgen.operators.generate.gates import GATES

            if output_gate not in GATES:
                raise ValueError(f"output_gate_not_registered:{output_gate}")
            gate = GATES[output_gate]
            base_generate = self.generator.generate

            async def _gated_generate(batch: list) -> list[dict]:
                return gate(await base_generate(batch))

            self.generator.generate = _gated_generate

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
        for input_trace_id, qa_pairs in zip(
            [item["_trace_id"] for item in batch], results
        ):
            if not qa_pairs:
                continue
            for qa_pair in qa_pairs:
                res = self.generator.format_generation_results(
                    qa_pair, output_data_format=self.data_format
                )
                res["_trace_id"] = self.get_trace_id(res)
                final_results.append(res)
                meta_updates.setdefault(input_trace_id, []).append(res["_trace_id"])
        return final_results, meta_updates
