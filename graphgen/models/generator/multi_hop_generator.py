import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.models.generator.support import parse_support
from graphgen.utils import detect_main_language, logger


class MultiHopGenerator(BaseGenerator):
    TEMPLATE_KEY = "multi_hop"

    def build_prompt(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
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
        language = detect_main_language(entities_str + relationships_str)
        prompt = self.template(language).format(
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
        qa_pairs = [{"question": question, "answer": answer}]
        support = parse_support(response)
        if support:
            qa_pairs[0]["support"] = support
        return qa_pairs
