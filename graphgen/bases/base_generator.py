from abc import ABC, abstractmethod
from typing import Any

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper


class BaseGenerator(ABC):
    """
    Generate QAs based on given prompts.
    """

    def __init__(self, llm_client: BaseLLMWrapper):
        self.llm_client = llm_client

    @staticmethod
    @abstractmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """Build prompt for LLM based on the given batch"""

    @staticmethod
    @abstractmethod
    def parse_response(response: str) -> Any:
        """Parse the LLM response and return the generated QAs"""

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> dict[str, Any]:
        """
        Generate QAs based on a given batch.
        :param batch
        :return: QA pairs
        """
        result = {}
        prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(prompt)
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        result.update(qa_pairs)
        return result

    @staticmethod
    def format_generation_results(
        results: list[dict], output_data_format: str
    ) -> list[dict[str, Any]]:

        flat_results = []
        for item in results:
            for _, qa_data in item.items():
                question = qa_data.get("question", "")
                answer = qa_data.get("answer", "")
                if "options" in qa_data and qa_data["options"]:
                    options = qa_data["options"]
                    options_str = "\n".join(
                        [f"{key}. {options[key]}" for key in sorted(options.keys())]
                    )
                    question += f"\nOptions:\n{options_str}"

                if output_data_format == "Alpaca":
                    flat_results.append(
                        {
                            "instruction": question,
                            "input": "",
                            "output": answer,
                        }
                    )
                elif output_data_format == "Sharegpt":
                    flat_results.append(
                        {
                            "conversations": [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ]
                        }
                    )
                elif output_data_format == "ChatML":
                    flat_results.append(
                        {
                            "messages": [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer},
                            ]
                        }
                    )
                    )
                else:
                    raise ValueError(
                        f"Unknown output data format: {output_data_format}"
                    )
        return flat_results
