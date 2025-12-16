import re
from typing import Any, Optional

from graphgen.bases import BaseGenerator
from graphgen.templates import OMICS_QA_GENERATION_PROMPT
from graphgen.utils import compute_content_hash, detect_main_language, logger


class OmicsQAGenerator(BaseGenerator):
    """
    Unified QA generator for multi-omics data (DNA, RNA, Protein).
    Automatically extracts and attaches molecule-specific caption information to QA pairs.
    """

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
        prompt = OMICS_QA_GENERATION_PROMPT[language].format(
            entities=entities_str, relationships=relationships_str
        )
        return prompt

    @staticmethod
    def parse_response(response: str) -> Any:
        """
        Parse the LLM response and return the generated QAs
        :param response
        :return: QA pairs
        """
        qa_pairs = {}
        qa_list = response.strip().split("\n\n")
        for qa in qa_list:
            match = re.search(
                r"Question:\s*(.*?)\s*Answer:\s*(.*)", qa, re.DOTALL
            ) or re.search(r"问题：\s*(.*?)\s*答案：\s*(.*)", qa, re.DOTALL)

            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
            else:
                if qa:
                    logger.error("Failed to parse QA pair: %s", qa)
                continue
            question = question.strip('"')
            answer = answer.strip('"')
            logger.debug("Question: %s", question)
            logger.debug("Answer: %s", answer)
            qa_pairs[compute_content_hash(question)] = {
                "question": question,
                "answer": answer,
            }
        return qa_pairs

    @staticmethod
    def _extract_caption(node_data: dict, molecule_type: str) -> Optional[dict]:
        """
        Extract molecule-specific caption information from node data.
        
        :param node_data: Node data dictionary
        :param molecule_type: Type of molecule ("dna", "rna", or "protein")
        :return: Caption dictionary or None
        """
        molecule_type_lower = molecule_type.lower()
        
        # Check if there's already a caption field (e.g., protein_caption, dna_caption, rna_caption)
        caption_key = f"{molecule_type_lower}_caption"
        if caption_key in node_data and node_data[caption_key]:
            if isinstance(node_data[caption_key], list) and len(node_data[caption_key]) > 0:
                return node_data[caption_key][0] if isinstance(node_data[caption_key][0], dict) else node_data[caption_key]
            elif isinstance(node_data[caption_key], dict):
                return node_data[caption_key]
        
        # Extract from metadata or node data based on molecule type
        caption = {}
        
        if molecule_type_lower == "protein":
            # Extract protein-specific fields
            if "protein" in node_data and node_data["protein"]:
                if isinstance(node_data["protein"], list) and len(node_data["protein"]) > 0:
                    return node_data["protein"][0] if isinstance(node_data["protein"][0], dict) else node_data["protein"]
                elif isinstance(node_data["protein"], dict):
                    return node_data["protein"]
            
            # Fallback: extract from node data fields
            caption_fields = ["protein_name", "gene_names", "organism", "function", "sequence", "id", "database"]
            for field in caption_fields:
                if field in node_data:
                    caption[field] = node_data[field]
        
        elif molecule_type_lower == "dna":
            # Extract DNA-specific fields
            caption_fields = [
                "gene_name", "gene_description", "organism", "chromosome", 
                "genomic_location", "function", "gene_type", "id", "database"
            ]
            for field in caption_fields:
                if field in node_data:
                    caption[field] = node_data[field]
        
        elif molecule_type_lower == "rna":
            # Extract RNA-specific fields
            caption_fields = [
                "rna_type", "description", "organism", "related_genes", 
                "gene_name", "so_term", "id", "database", "rnacentral_id"
            ]
            for field in caption_fields:
                if field in node_data:
                    caption[field] = node_data[field]
        
        return caption if caption else None

    @staticmethod
    def _detect_molecule_type(nodes: list[tuple[str, dict]]) -> str:
        """
        Detect molecule type from nodes.
        Priority: Check node type, then check metadata, then check node data fields.
        
        :param nodes: List of (node_id, node_data) tuples
        :return: Detected molecule type ("dna", "rna", "protein", or "unknown")
        """
        for _, node_data in nodes:
            # Check node type field
            node_type = node_data.get("type", "").lower()
            if node_type in ("dna", "rna", "protein"):
                return node_type
            
            # Check molecule_type in metadata or node data
            molecule_type = node_data.get("molecule_type", "").lower()
            if molecule_type in ("dna", "rna", "protein"):
                return molecule_type
            
            # Check for type-specific fields
            if "protein" in node_data or "protein_name" in node_data or "protein_caption" in node_data:
                return "protein"
            if "gene_name" in node_data and "chromosome" in node_data:
                return "dna"
            if "rna_type" in node_data or "rnacentral_id" in node_data:
                return "rna"
        
        return "unknown"

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> dict[str, Any]:
        """
        Generate QAs based on a given batch.
        Automatically extracts and attaches molecule-specific caption information.
        
        :param batch
        :return: QA pairs with attached molecule captions
        """
        result = {}
        prompt = self.build_prompt(batch)
        response = await self.llm_client.generate_answer(prompt)
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        
        nodes, _ = batch
        
        # Detect molecule type from nodes
        molecule_type = self._detect_molecule_type(nodes)
        
        # Extract caption for each node and attach to QA pairs
        # Only attach caption once per batch (from the first relevant node)
        caption_attached = False
        for node in nodes:
            node_data = node[1]
            caption = self._extract_caption(node_data, molecule_type)
            
            if caption and not caption_attached:
                # Attach caption to all QA pairs
                for qa in qa_pairs.values():
                    # Use molecule_type as the key (dna, rna, or protein)
                    qa[molecule_type] = caption
                caption_attached = True
                break  # Only need to attach once per batch
        
        result.update(qa_pairs)
        return result

    @staticmethod
    def format_generation_results(
        results: list[dict], output_data_format: str
    ) -> list[dict[str, Any]]:
        """
        Format generation results with molecule-specific caption fields.
        Supports dna, rna, and protein fields in output.
        """
        if output_data_format == "Alpaca":
            results = [
                {
                    "instruction": v["question"],
                    "input": "",
                    "output": v["answer"],
                    "dna": v.get("dna", ""),
                    "rna": v.get("rna", ""),
                    "protein": v.get("protein", ""),
                }
                for item in results
                for k, v in item.items()
            ]
        elif output_data_format == "Sharegpt":
            results = [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": [
                                {
                                    "text": v["question"],
                                    "dna": v.get("dna", ""),
                                    "rna": v.get("rna", ""),
                                    "protein": v.get("protein", ""),
                                }
                            ],
                        },
                        {"from": "gpt", "value": v["answer"]},
                    ]
                }
                for item in results
                for k, v in item.items()
            ]
        elif output_data_format == "ChatML":
            results = [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": v["question"],
                                    "dna": v.get("dna", ""),
                                    "rna": v.get("rna", ""),
                                    "protein": v.get("protein", ""),
                                }
                            ],
                        },
                        {"role": "assistant", "content": v["answer"]},
                    ]
                }
                for item in results
                for k, v in item.items()
            ]
        else:
            raise ValueError(f"Unknown output data format: {output_data_format}")
        return results
