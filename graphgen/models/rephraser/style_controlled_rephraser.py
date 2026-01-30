from typing import Any, Optional

from graphgen.bases import BaseRephraser
from graphgen.templates import STYLE_CONTROLLED_REPHRASING_PROMPTS
from graphgen.utils import detect_main_language, logger


class StyleControlledRephraser(BaseRephraser):
    """
    Style Controlled Rephraser rephrases the input text based on a specified style.
    """

    def __init__(self, llm_client: Any, style: str = "critical_analysis"):
        super().__init__(llm_client)
        self.style = style

    def build_prompt(self, text: str) -> str:
        logger.debug("Text to be rephrased: %s", text)
        language = detect_main_language(text)
        prompt_template = STYLE_CONTROLLED_REPHRASING_PROMPTS[self.style][language]
        prompt = prompt_template.format(text=text)
        return prompt

    @staticmethod
    def parse_response(response: str) -> Optional[dict]:
        result = response.strip()
        logger.debug("Raw rephrased response: %s", result)
        if not result:
            return None
        return {
            "content": result,
        }
