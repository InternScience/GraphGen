import asyncio
import logging

from graphgen.bases.datatypes import Chunk
from graphgen.models.generator.vqa_generator import VQAGenerator
from graphgen.models.kg_builder.light_rag_kg_builder import LightRAGKGBuilder
from graphgen.utils.log import CURRENT_LOGGER_VAR


class _DummyTokenizer:
    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


class _DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tokenizer = _DummyTokenizer()

    async def generate_answer(self, *args, **kwargs):
        return self.responses.pop(0)


def test_light_rag_filters_entities_and_relations_without_grounded_evidence():
    llm = _DummyLLM(
        [
            (
                '("entity"<|>"Alpha"<|>"concept"<|>"Alpha summary"<|>"Alpha is present")##'
                '("entity"<|>"Ghost"<|>"concept"<|>"Ghost summary"<|>"Ghost evidence")##'
                '("relationship"<|>"Alpha"<|>"Ghost"<|>"related_to"<|>"unsupported link"<|>"Ghost evidence"<|>0.9)'
                "<|COMPLETE|>"
            ),
            "no",
        ]
    )
    builder = LightRAGKGBuilder(
        llm_client=llm,
        require_entity_evidence=True,
        require_relation_evidence=True,
        validate_evidence_in_source=True,
    )
    token = CURRENT_LOGGER_VAR.set(logging.getLogger("test-evidence"))

    try:
        nodes, edges = asyncio.run(
            builder.extract(
                Chunk(
                    id="chunk-1",
                    type="text",
                    content="Alpha is present in the source text.",
                    metadata={},
                )
            )
        )
    finally:
        CURRENT_LOGGER_VAR.reset(token)

    assert set(nodes.keys()) == {"ALPHA"}
    assert edges == {}
    assert nodes["ALPHA"][0]["evidence_span"] == "Alpha is present"


def test_vqa_prompt_includes_grounding_evidence():
    prompt = VQAGenerator.build_prompt(
        (
            [
                (
                    "FIGURE-1",
                    {
                        "description": "A microscopy image of treated tissue.",
                        "evidence_span": "Figure 1 shows treated tissue.",
                        "metadata": '{"img_path":"demo.png"}',
                    },
                )
            ],
            [
                (
                    "FIGURE-1",
                    "LATENCY",
                    {
                        "description": "The figure reports a 12 ms latency.",
                        "relation_type": "has_latency",
                        "evidence_span": "Latency is 12 ms.",
                    },
                )
            ],
        )
    )

    assert "Evidence: Figure 1 shows treated tissue." in prompt
    assert "Evidence: Latency is 12 ms." in prompt
    assert "[has_latency]" in prompt
