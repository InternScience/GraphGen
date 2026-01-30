import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import COT_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger


class CoTGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """
        Build prompts for COT Template Design.
        :param batch:
        :return:
        """
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
        prompt = COT_GENERATION_PROMPT[language]["COT_TEMPLATE_DESIGN"].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def build_prompt_for_cot_generation(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        question: str,
        reasoning_path: str,
    ) -> str:
        """
        Build prompts for COT Generation.
        """
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
        prompt = COT_GENERATION_PROMPT[language]["COT_GENERATION"].format(
            entities=entities_str,
            relationships=relationships_str,
            question=question,
            reasoning_template=reasoning_path,
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> dict:
        """
        Parse CoT template from response.
        :param response:
        :return: dict with question and reasoning_path
        """
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        reasoning_path_match = re.search(
            r"<reasoning_path>(.*?)</reasoning_path>", response, re.DOTALL
        )

        if question_match and reasoning_path_match:
            question = question_match.group(1).strip()
            reasoning_path = reasoning_path_match.group(1).strip()
        else:
            logger.warning("Failed to parse response: %s", response)
            return {}

        question = question.strip('"').strip("'")
        reasoning_path = reasoning_path.strip('"').strip("'")

        logger.debug("CoT Question: %s", question)
        logger.debug("CoT Reasoning Path: %s", reasoning_path)
        return {
            "question": question,
            "reasoning_path": reasoning_path,
        }

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
        response = self.parse_response(response)
        if not response:
            return []
        question, reasoning_path = response["question"], response["reasoning_path"]
        prompt = self.build_prompt_for_cot_generation(batch, question, reasoning_path)
        cot_answer = await self.llm_client.generate_answer(prompt)
        logger.debug("CoT Answer: %s", cot_answer)
        return [
            {
                "question": question,
                "answer": cot_answer,
                "reasoning_path": reasoning_path,
            }
        ]
