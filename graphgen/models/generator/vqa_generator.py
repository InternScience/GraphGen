import json
import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import VQA_GENERATION_PROMPT
from graphgen.utils import detect_main_language, logger

from .context_utils import build_grounded_context


class VQAGenerator(BaseGenerator):
    def __init__(self, llm_client):
        super().__init__(llm_client)

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        entities_str, relationships_str = build_grounded_context(batch)
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _build_context_keywords(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> set[str]:
        nodes, edges = batch
        raw_text = []
        raw_text.extend([node[0] for node in nodes])
        raw_text.extend([node[1].get("description", "") for node in nodes])
        raw_text.extend([str(edge[0]) for edge in edges])
        raw_text.extend([str(edge[1]) for edge in edges])
        raw_text.extend([edge[2].get("description", "") for edge in edges])

        keyword_pattern = re.compile(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9_\-/]{2,}")
        return {token.lower() for token in keyword_pattern.findall("\n".join(raw_text))}

    def _is_high_quality_qa(
        self, qa_pair: dict, context_keywords: set[str], seen_pairs: set[str]
    ) -> bool:
        question = qa_pair.get("question", "").strip()
        answer = qa_pair.get("answer", "").strip()
        if not question or not answer:
            return False

        if any(token in question.lower() for token in ["todo", "placeholder", "n/a"]):
            return False
        if any(
            token in answer.lower()
            for token in ["i don't know", "unknown", "无法确定", "不确定"]
        ):
            return False

        normalized_signature = (
            f"{self._normalize_text(question)}|{self._normalize_text(answer)}"
        )
        if normalized_signature in seen_pairs:
            return False

        context_hits = [
            keyword
            for keyword in context_keywords
            if keyword in question.lower() or keyword in answer.lower()
        ]
        if not context_hits:
            return False

        seen_pairs.add(normalized_signature)
        return True

    @staticmethod
    def _extract_img_path(nodes: list[tuple[str, dict]]) -> str:
        for node in nodes:
            node_data = node[1]
            if "metadata" not in node_data or not node_data["metadata"]:
                continue
            try:
                raw_metadata = json.loads(node_data["metadata"])
            except (json.JSONDecodeError, TypeError):
                continue
            metadata = (
                raw_metadata.get("metadata", {})
                if isinstance(raw_metadata.get("metadata"), dict)
                else raw_metadata
            )
            img_path = metadata.get("img_path") or metadata.get("path", "")
            if img_path:
                return img_path
        return ""

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
        context_keywords = self._build_context_keywords(batch)
        seen_pairs = set()
        filtered_pairs = [
            qa
            for qa in qa_pairs
            if self._is_high_quality_qa(
                qa, context_keywords=context_keywords, seen_pairs=seen_pairs
            )
        ]

        if len(filtered_pairs) < len(qa_pairs):
            logger.info(
                "VQA quality filter removed %d of %d QA pairs",
                len(qa_pairs) - len(filtered_pairs),
                len(qa_pairs),
            )

        img_path = self._extract_img_path(nodes)
        for qa in filtered_pairs:
            qa["img_path"] = img_path

        return filtered_pairs

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        question = result.get("question", "")
        answer = result.get("answer", "")
        img_path = result.get("img_path", "")
        if output_data_format == "Alpaca":
            result = {
                "instruction": question,
                "input": "",
                "output": answer,
            }
            if img_path:
                result["image"] = img_path
            return result
        if output_data_format == "Sharegpt":
            user_value = [{"text": question}]
            if img_path:
                user_value[0]["image"] = img_path
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": user_value,
                    },
                    {"from": "gpt", "value": [{"text": answer}]},
                ]
            }
        if output_data_format == "ChatML":
            user_content = [{"text": question}]
            if img_path:
                user_content[0]["image"] = img_path
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ]
            }
        raise ValueError(f"Unknown output data format: {output_data_format}")
