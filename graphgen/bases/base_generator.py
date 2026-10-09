from abc import ABC, abstractmethod
from typing import Any, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper


class BaseGenerator(ABC):
    """
    Generate QAs based on given prompts.

    PMS fork: prompt templates can be overridden per instance via a versioned
    prompt profile (see graphgen.templates.prompt_profiles). Generators declare
    their official template under ``TEMPLATE_KEY`` and render through
    ``self.template(...)``; without a profile the official constant is used
    byte-for-byte.
    """

    #: key into graphgen.templates.prompt_profiles.OFFICIAL_TEMPLATES
    TEMPLATE_KEY: Optional[str] = None

    def __init__(self, llm_client: BaseLLMWrapper):
        self.llm_client = llm_client
        self._profile_templates: Optional[dict] = None

    def apply_prompt_profile(self, profile: str) -> None:
        """Attach a versioned prompt profile to this generator instance."""
        from graphgen.templates.prompt_profiles import load_profile_template

        if self.TEMPLATE_KEY is None:
            raise ValueError(
                f"prompt_profile_unsupported_generator:{type(self).__name__}"
            )
        self._profile_templates = load_profile_template(profile, self.TEMPLATE_KEY)

    def template(self, language: str, key: Optional[str] = None) -> str:
        """Resolve the effective template for this generator.

        :param language: detected language key ("en" / "zh").
        :param key: nested sub-template key for multi-phase generators.
        """
        from graphgen.templates.prompt_profiles import OFFICIAL_TEMPLATES

        if self.TEMPLATE_KEY is None:
            raise ValueError(
                f"prompt_template_key_missing:{type(self).__name__}"
            )
        if self._profile_templates is not None:
            node = self._profile_templates[language]
        else:
            node = OFFICIAL_TEMPLATES[self.TEMPLATE_KEY][language]
        return node[key] if key is not None else node

    @staticmethod
    @abstractmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """Build prompt for LLM based on the given batch"""

    @staticmethod
    @abstractmethod
    def parse_response(response: str) -> list[dict]:
        """Parse the LLM response and return the generated QAs"""

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> list[dict]:
        """
        Generate QAs based on a given batch.
        :param batch
        :return: QA pairs
        """
        prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(prompt)
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        return qa_pairs

    @staticmethod
    def format_generation_results(
        result: dict, output_data_format: str
    ) -> dict[str, Any]:
        question = result.get("question", "")
        answer = result.get("answer", "")
        # PMS fork（Phase 2）：已校验的 <support> 引用块随 QA_pairs 输出透传。
        support = result.get("support")
        if "options" in result and result["options"]:
            options = result["options"]
            options_str = "\n".join(
                [f"{key}. {options[key]}" for key in sorted(options.keys())]
            )
            question += f"\nOptions:\n{options_str}"

        if output_data_format == "Alpaca":
            return {
                "instruction": question,
                "input": "",
                "output": answer,
            }

        if output_data_format == "Sharegpt":
            return {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]
            }
        if output_data_format == "ChatML":
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }

        if output_data_format == "QA_pairs":
            output = {
                "question": question,
                "answer": answer,
            }
            if support:
                output["support"] = support
            return output
        raise ValueError(f"Unknown output data format: {output_data_format}")
