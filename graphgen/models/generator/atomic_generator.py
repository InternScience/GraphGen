import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import ATOMIC_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger

from .context_utils import build_grounded_context


class AtomicGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        entities_str, relationships_str = build_grounded_context(batch)
        context = entities_str
        if relationships_str:
            context = f"{context}\n{relationships_str}".strip()
        language = detect_main_language(context)

        prompt = ATOMIC_GENERATION_PROMPT[language].format(context=context)
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        """
        AtomicGenerator normally generates one QA pair per response.
        So we just need to parse one QA pair from the response.
        :param response:
        :return:
        """
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
