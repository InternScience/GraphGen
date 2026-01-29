import re
from typing import Any, Optional

from graphgen.bases import BaseGenerator
from graphgen.templates import AGGREGATED_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger


class AggregatedGenerator(BaseGenerator):
    """
    Aggregated Generator follows a TWO-STEP process:
    1. rephrase: Rephrase the input nodes and edges into a coherent text that maintains the original meaning.
                 The rephrased text is considered as answer to be used in the next step.
    2. question generation: Generate relevant questions based on the rephrased text.
    """

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """
        Build prompts for REPHRASE.
        :param batch
        :return:
        """
        nodes, edges = batch
        entities_str = "\n".join(
            [
                f"{index + 1}. {node[0]}: {node[1]['description']}"
                for index, node in enumerate(nodes)
            ]
        )
        relations_str = "\n".join(
            [
                f"{index + 1}. {edge[0]} -- {edge[1]}: {edge[2]['description']}"
                for index, edge in enumerate(edges)
            ]
        )
        language = detect_main_language(entities_str + relations_str)

        # TODO: configure add_context
        #     if add_context:
        #         original_ids = [
        #             node["source_id"].split("<SEP>")[0] for node in _process_nodes
        #         ] + [edge[2]["source_id"].split("<SEP>")[0] for edge in _process_edges]
        #         original_ids = list(set(original_ids))
        #         original_text = await text_chunks_storage.get_by_ids(original_ids)
        #         original_text = "\n".join(
        #             [
        #                 f"{index + 1}. {text['content']}"
        #                 for index, text in enumerate(original_text)
        #             ]
        #         )
        prompt = AGGREGATED_GENERATION_PROMPT[language]["ANSWER_REPHRASING"].format(
            entities=entities_str, relationships=relations_str
        )
        return prompt

    @staticmethod
    def parse_rephrased_text(response: str) -> Optional[str]:
        """
        Parse the rephrased text from the response.
        :param response:
        :return: rephrased text
        """
        rephrased_match = re.search(
            r"<rephrased_text>(.*?)</rephrased_text>", response, re.DOTALL
        )
        if rephrased_match:
            rephrased_text = rephrased_match.group(1).strip()
        else:
            logger.warning("Failed to parse rephrased text from response: %s", response)
            return None
        return rephrased_text.strip('"').strip("'")

    @staticmethod
    def _build_prompt_for_question_generation(answer: str) -> str:
        """
        Build prompts for QUESTION GENERATION.
        :param answer:
        :return:
        """
        language = detect_main_language(answer)
        prompt = AGGREGATED_GENERATION_PROMPT[language]["QUESTION_GENERATION"].format(
            answer=answer
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> dict:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
        else:
            logger.warning("Failed to parse question from response: %s", response)
            return {"question": ""}
        return {"question": question.strip('"').strip("'")}

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
        rephrasing_prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(rephrasing_prompt)
        context = self.parse_rephrased_text(response)
        if not context:
            return []
        question_generation_prompt = self._build_prompt_for_question_generation(context)
        response = await self.llm_client.generate_answer(question_generation_prompt)
        question = self.parse_response(response)["question"]
        if not question:
            return []
        logger.debug("Question: %s", question)
        logger.debug("Answer: %s", context)
        qa_pairs = {
            "question": question,
            "answer": context,
        }
        return [qa_pairs]
