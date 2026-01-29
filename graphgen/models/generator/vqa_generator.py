import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import VQA_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger


class VQAGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
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
        prompt = VQA_GENERATION_PROMPT[language].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        """
        Parse the LLM response and return the generated QAs
        :param response
        :return: QA pairs
        """
        qa_pairs = []
        pattern = r"<question>(.*?)</question>\s*<answer>(.*?)</answer>"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            for question, answer in matches:
                question = question.strip().strip('"').strip("'")
                answer = answer.strip().strip('"').strip("'")
                logger.debug("Question: %s", question)
                logger.debug("Answer: %s", answer)
                qa_pairs.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )
        else:
            logger.warning("Error parsing the response %s", response)
        return qa_pairs

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
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        nodes, _ = batch
        for node in nodes:
            node_data = node[1]
            if "image_data" in node_data and node_data["image_data"]:
                img_path = node_data["image_data"]["img_path"]
                for qa in qa_pairs:
                    qa["img_path"] = img_path
        return qa_pairs

    @staticmethod
    def format_generation_results(
        result: list[dict], output_data_format: str
    ) -> list[dict[str, Any]]:
        if output_data_format == "Alpaca":
            result = [
                {
                    "instruction": v["question"],
                    "input": "",
                    "output": v["answer"],
                    "image": v.get("img_path", ""),
                }
                for item in result
                for k, v in item.items()
            ]
        elif output_data_format == "Sharegpt":
            result = [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": [
                                {"text": v["question"], "image": v.get("img_path", "")}
                            ],
                        },
                        {"from": "gpt", "value": [{"text": v["answer"]}]},
                    ]
                }
                for item in result
                for k, v in item.items()
            ]
        elif output_data_format == "ChatML":
            result = [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": v["question"], "image": v.get("img_path", "")}
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": v["answer"]}],
                        },
                    ]
                }
                for item in result
                for k, v in item.items()
            ]
        else:
            raise ValueError(f"Unknown output data format: {output_data_format}")
        return result
