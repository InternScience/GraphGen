import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import MAQ_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class MultiAnswerGenerator(BaseGenerator):
    def __init__(self, llm_client, num_of_questions) -> None:
        super().__init__(llm_client)
        self.num_of_questions = num_of_questions

    @staticmethod
    def parse_response(response: str) -> Any:
        """
        Parse multiple-answer QA pairs from the LLM response.
        Each QA pair contains question text, four options, and the correct answers (one or more).

        :param response: The LLM response containing XML-formatted QA pairs
        :return: Dictionary mapping question hash to question data, where each
                 value is a dict with "question", "options", "answer", and
                 "correct_answer_text" keys
        """
        qa_pairs = {}

        # Extract all QA pair blocks
        qa_blocks = re.findall(r"<qa_pair>(.*?)</qa_pair>", response, re.DOTALL)

        if not qa_blocks:
            logger.warning("No QA pairs found in response: %s", response)
            return {}

        for block in qa_blocks:
            # Extract and clean question text
            q_match = re.search(r"<question>(.*?)</question>", block, re.DOTALL)
            if not q_match:
                logger.warning("Failed to parse question from block: %s", block)
                continue
            question = q_match.group(1).strip().strip('"').strip("'")

            # Extract and parse options (A, B, C, D)
            opt_match = re.search(r"<options>(.*?)</options>", block, re.DOTALL)
            if not opt_match:
                logger.warning("Failed to parse options from block: %s", block)
                continue

            options = {}
            options_text = opt_match.group(1).strip()
            for line in options_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match patterns like "A. text" or "B. text"
                if m := re.match(r"^([A-Z])[.\s]\s*(.*)$", line):
                    letter, text = m.groups()
                    options[letter] = text.strip()

            # Extract and validate answer
            ans_match = re.search(r"<answer>(.*?)</answer>", block, re.DOTALL)
            if not ans_match:
                logger.warning("Failed to parse answer from block: %s", block)
                continue
            answer_text = ans_match.group(1).strip().strip('"').strip("'")
            answers = [ans.strip().upper() for ans in answer_text.split(",") if ans.strip()]
            invalid_answers = [ans for ans in answers if ans not in options]
            if invalid_answers:
                logger.warning(
                    "Answers %s not found in options: %s",
                    invalid_answers,
                    list(options.keys()),
                )
                continue

            # Ensure at least one valid answer
            if len(answers) == 0:
                logger.warning("No valid answers found in: %s", answer_text)
                continue

            # Build result entry with question hash as key
            question_hash = compute_content_hash(question)
            qa_pairs[question_hash] = {
                "question": question,
                "options": options,  # Dict like {"A": "text", "B": "text", ...}
                "answer": ", ".join(answers),
            }

            logger.debug("Successfully parsed MAQ: %s", question[:50])

        if not qa_pairs:
            logger.error("Failed to parse any valid MAQ pairs from response")

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
        prompt = MAQ_GENERATION_PROMPT[language].format(
            context=context,
            num_of_questions=self.num_of_questions,
        )
        return prompt
