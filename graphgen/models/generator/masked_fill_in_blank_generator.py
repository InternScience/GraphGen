import re
import random
from typing import Any, Optional

from graphgen.bases import BaseGenerator
from graphgen.templates import AGGREGATED_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger

random.seed(42)

class MaskedFillInBlankGenerator(BaseGenerator):
    """
    Masked Fill-in-blank Generator follows a TWO-STEP process:
    1. rephrase: Rephrase the input nodes and edges into a coherent text that maintains the original meaning.
    2. mask: Randomly select a node from the input nodes, and then mask the name of the node in the rephrased text.
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
    def parse_response(response: str) -> dict:
        pass

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
        
        nodes, edge = batch
        assert len(nodes) == 2, "MaskedFillInBlankGenerator currently only supports triples, which should has 2 nodes."
        assert len(edge) == 1, "MaskedFillInBlankGenerator currently only supports triples, which should has 1 edge."

        node1, node2 = nodes
        mask_node = random.choice([node1, node2])
        mask_node_name = mask_node[1]["entity_name"].strip('\'" \n\r\t')

        mask_pattern = re.compile(re.escape(mask_node_name), re.IGNORECASE)
        masked_context = mask_pattern.sub("___", context)
        # For accuracy, extract the actual replaced text from the context as the ground truth (keeping the original case)
        gth = re.search(mask_pattern, context).group(0)

        logger.debug("masked_context: %s", masked_context)
        qa_pairs = {
            "question": masked_context,
            "answer": gth,
        }
        return [qa_pairs]

