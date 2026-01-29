import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import FILL_IN_BLANK_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class FillInBlankGenerator(BaseGenerator):
    def __init__(self, llm_client, num_of_questions) -> None:
        super().__init__(llm_client)
        self.num_of_questions = num_of_questions

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        """
        Parse fill-in-the-blank QA pairs from the LLM response.
        Each QA pair contains question text with placeholders and the correct answer(s).

        :param response: The LLM response containing XML-formatted QA pairs
        :return: Dictionary mapping question hash to question data, where each
                 value is a dict with "question", "answer", and "answers" keys
        """
        qa_pairs = []

        # Extract all QA pair blocks
        qa_blocks = re.findall(r"<qa_pair>(.*?)</qa_pair>", response, re.DOTALL)

        if not qa_blocks:
            logger.warning("No QA pairs found in response: %s", response)
            return qa_pairs

        for block in qa_blocks:
            # Extract and clean question text
            q_match = re.search(r"<question>(.*?)</question>", block, re.DOTALL)
            if not q_match:
                logger.warning("Failed to parse question from block: %s", block)
                continue
            question = q_match.group(1).strip().strip('"').strip("'")

            # Extract and clean answer text
            ans_match = re.search(r"<answer>(.*?)</answer>", block, re.DOTALL)
            if not ans_match:
                logger.warning("Failed to parse answer from block: %s", block)
                continue

            answer_text = ans_match.group(1).strip().strip('"').strip("'")

            # Parse multiple answers (e.g., "A8X, 八百万" or "A8X")
            # Split by comma and strip whitespace from each answer
            answers = [ans.strip() for ans in answer_text.split(",") if ans.strip()]

            # Ensure at least one valid answer
            if len(answers) == 0:
                logger.warning("No valid answers found in: %s", answer_text)
                continue

            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer_text,  # Original answer text with commas
                    "answers": answers,  # List of individual answers: ["A8X"] or ["A8X", "八百万"]
                }
            )

            logger.debug(
                "Successfully parsed fill-in-the-blank question: %s", question[:50]
            )

        if not qa_pairs:
            logger.error("Failed to parse any valid QA pairs from response")

        return qa_pairs

    # pylint: disable=W0221
    def build_prompt(
        self, batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, edges = batch
        entities_str = "\n".join(
            [
                f"{index + 1}. {node[0]}: {node[1]['description']}"
                for index, node in enumerate(nodes)
            ]
        )

        relationships_str = "\n".join(
            [
                f"{index + 1}. {edge[0]} -- {edge[1]}: {edge[2]['description']}"
                for index, edge in enumerate(edges)
            ]
        )
        context = entities_str + "\n" + relationships_str
        language = detect_main_language(entities_str + relationships_str)
        prompt = FILL_IN_BLANK_GENERATION_PROMPT[language].format(
            context=context,
            num_of_questions=self.num_of_questions,
        )
        return prompt
