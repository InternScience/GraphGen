from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import AGGREGATED_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger


class QuizGenerator(BaseGenerator):
    """
    Quiz Generator rephrases given descriptions to create quiz questions.
    """

    @staticmethod
    def build_prompt(description: str) -> str:
        """
        Build prompt for rephrasing the description.
        :param description:
        :return:
        """
        language = detect_main_language(description)
        prompt = AGGREGATED_GENERATION_PROMPT[language][
            "DESCRIPTION_REPHRASING"
        ].format(description=description)
        return prompt

    @staticmethod
    def parse_rephrased_text(response: str) -> str:
        """
        Parse the rephrased text from the response.
        :param response:
        :return:
        """
        rephrased_text = response.strip().strip('"')
        logger.debug("Rephrased Text: %s", rephrased_text)
        return rephrased_text

    @staticmethod
    def parse_response(response: str) -> Any:
        pass
