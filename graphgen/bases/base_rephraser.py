from abc import ABC, abstractmethod
from typing import Any

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper


class BaseRephraser(ABC):
    """
    Rephrase text based on given prompts.
    """

    def __init__(self, llm_client: BaseLLMWrapper):
        self.llm_client = llm_client

    @abstractmethod
    def build_prompt(self, text: str) -> str:
        """Build prompt for LLM based on the given text"""

    @staticmethod
    @abstractmethod
    def parse_response(response: str) -> Any:
        """Parse the LLM response and return the rephrased text"""

    async def rephrase(
        self,
        item: dict,
    ) -> dict:
        text = item["content"]
        prompt = self.build_prompt(text)
        response = await self.llm_client.generate_answer(prompt)
        return self.parse_response(response)
