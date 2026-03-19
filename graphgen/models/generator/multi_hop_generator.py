import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import MULTI_HOP_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger

from .context_utils import build_grounded_context


class MultiHopGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        entities_str, relationships_str = build_grounded_context(batch)
        language = detect_main_language(entities_str + relationships_str)
        prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()
        else:
            logger.warning("Failed to parse response: %s", response)
            return []

        question = question.strip('"').strip("'")
        answer = answer.strip('"').strip("'")
        logger.debug("Question: %s", question)
        logger.debug("Answer: %s", answer)
        return [{"question": question, "answer": answer}]
