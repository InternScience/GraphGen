from typing import Tuple

from graphgen.bases import BaseKVStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger, run_concurrent


def _gated_generate(generate, gate):
    """Wrap an async generate so its parsed QA list passes through ``gate``."""

    async def _wrapper(batch: list) -> list[dict]:
        return gate(await generate(batch))

    return _wrapper


#: method -> (generator class name, default num_of_questions or None)
_GENERATOR_SPECS = {
    "atomic": ("AtomicGenerator", None),
    "aggregated": ("AggregatedGenerator", None),
    "multi_hop": ("MultiHopGenerator", None),
    "cot": ("CoTGenerator", None),
    "vqa": ("VQAGenerator", None),
    "multi_choice": ("MultiChoiceGenerator", 5),
    "multi_answer": ("MultiAnswerGenerator", 3),
    "fill_in_blank": ("FillInBlankGenerator", 5),
    "masked_fill_in_blank": ("MaskedFillInBlankGenerator", None),
    "true_false": ("TrueFalseGenerator", 5),
}


def _create_generator(llm_client: BaseLLMWrapper, method: str, generate_kwargs: dict):
    """Instantiate the generator selected by ``method``."""
    if method not in _GENERATOR_SPECS:
        raise ValueError(f"Unsupported generation mode: {method}")
    class_name, default_num_of_questions = _GENERATOR_SPECS[method]
    from graphgen import models as graphgen_models

    generator_cls = getattr(graphgen_models, class_name)
    if default_num_of_questions is None:
        return generator_cls(llm_client)
    return generator_cls(
        llm_client,
        num_of_questions=generate_kwargs.get(
            "num_of_questions", default_num_of_questions
        ),
    )


def _configure_generator(
    generator,
    prompt_profile: str | None,
    output_gate: str | None,
    llm_concurrency: int | None,
) -> None:
    """Apply optional profile / concurrency / gate configuration."""
    if prompt_profile:
        generator.apply_prompt_profile(prompt_profile)
    if llm_concurrency is not None:
        from graphgen.operators.generate.llm_concurrency import (
            bound_llm_concurrency,
        )

        generator.generate = bound_llm_concurrency(
            generator.generate, llm_concurrency
        )
    if output_gate:
        from graphgen.operators.generate.gates import GATES

        if output_gate not in GATES:
            raise ValueError(f"output_gate_not_registered:{output_gate}")
        generator.generate = _gated_generate(generator.generate, GATES[output_gate])


class GenerateService(BaseOperator):
    """
    Generate question-answer pairs based on nodes and edges.

    Optional additions (all default to official behavior when unset):
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
        self.generator = _create_generator(self.llm_client, method, generate_kwargs)
        _configure_generator(
            self.generator, prompt_profile, output_gate, llm_concurrency
        )

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
