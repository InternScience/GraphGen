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
    def _extract_caption(node_data: dict, molecule_type: str) -> Optional[dict]:  # pylint: disable=too-many-branches
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
                # Always return the first element if it's a dict, otherwise return None for consistency
                caption_val = node_data[caption_key][0]
                return caption_val if isinstance(caption_val, dict) else None
            if isinstance(node_data[caption_key], dict):
                return node_data[caption_key]

        # Field mappings for each molecule type
        field_mapping = {
            "protein": [
                "protein_name", "gene_names", "organism", "function",
                "sequence", "id", "database", "entry_name", "uniprot_id"
            ],
            "dna": [
                "gene_name", "gene_description", "organism", "chromosome",
                "genomic_location", "function", "gene_type", "id",
                "database", "sequence"
            ],
            "rna": [
                "rna_type", "description", "organism", "related_genes",
                "gene_name", "so_term", "id", "database",
                "rnacentral_id", "sequence"
            ],
        }

        # Extract fields based on molecule type
        caption = {}
        caption_fields = field_mapping.get(molecule_type_lower, [])
        for field in caption_fields:
            if field in node_data and node_data[field]:
                caption[field] = node_data[field]

        # Special handling for protein: check search results and existing protein field
        if molecule_type_lower == "protein":
            # Check for search result data (from UniProt search)
            if "_search_results" in node_data:
                search_results = node_data["_search_results"]
                if isinstance(search_results, list) and len(search_results) > 0:
                    first_result = search_results[0]
                    if isinstance(first_result, dict):
                        search_caption = {
                            "id": first_result.get("id", ""),
                            "protein_name": first_result.get("protein_name", ""),
                            "gene_names": first_result.get("gene_names", []),
                            "organism": first_result.get("organism", ""),
                            "function": first_result.get("function", []),
                            "sequence": node_data.get("sequence") or first_result.get("sequence", ""),
                            "database": "UniProt"
                        }
                        # Remove empty fields and return if any data exists
                        search_caption = {k: v for k, v in search_caption.items() if v}
                        if search_caption:
                            return search_caption

            # Merge with existing protein field if present
            if "protein" in node_data and node_data["protein"]:
                existing_protein = node_data["protein"]
                if isinstance(existing_protein, list) and len(existing_protein) > 0:
                    existing_protein = (
                        existing_protein[0]
                        if isinstance(existing_protein[0], dict)
                        else existing_protein
                    )
                if isinstance(existing_protein, dict):
                    for key, value in existing_protein.items():
                        if key not in caption and value:
                            caption[key] = value
                    # Ensure sequence from node_data takes precedence
                    if "sequence" in node_data and node_data["sequence"]:
                        caption["sequence"] = node_data["sequence"]

        # Fallback to description if no caption found
        if not caption and "description" in node_data:
            description = node_data["description"]
            if isinstance(description, str) and len(description) > 10:
                caption["description"] = description

        return caption if caption else None

    @staticmethod
    def _detect_molecule_type(nodes: list[tuple[str, dict]]) -> str:
        """
        Detect molecule type from nodes.
        Priority: Check node type, then check metadata, then check node data fields.

        :param nodes: List of (node_id, node_data) tuples
        :return: Detected molecule type ("dna", "rna", "protein", or "unknown")
        """
        if not nodes:
            return "unknown"

        # Type indicators for each molecule type
        type_indicators = {
            "protein": {
                "fields": ["protein_name", "uniprot_id", "entry_name", "protein_caption"],
                "source_prefix": "protein-",
                "description_keywords": ["protein"],
            },
            "dna": {
                "fields": ["gene_name", "chromosome", "genomic_location"],
                "source_prefix": "dna-",
                "description_keywords": ["gene", "dna", "chromosome"],
            },
            "rna": {
                "fields": ["rna_type", "rnacentral_id"],
                "source_prefix": "rna-",
                "description_keywords": ["rna", "transcript"],
            },
        }

        for _, node_data in nodes:
            # Priority 1: Check explicit type fields (most reliable)
            for field in ["type", "molecule_type"]:
                value = node_data.get(field, "").lower()
                if value in ("dna", "rna", "protein"):
                    return value

            # Priority 2: Check source_id prefix
            source_id = node_data.get("source_id", "").lower()
            for mol_type, indicators in type_indicators.items():
                if source_id.startswith(indicators["source_prefix"]):
                    return mol_type

            # Priority 3: Check type-specific fields
            for mol_type, indicators in type_indicators.items():
                if any(key in node_data for key in indicators["fields"]):
                    # Special check for DNA: need chromosome or genomic_location
                    if mol_type == "dna" and not any(key in node_data for key in ["chromosome", "genomic_location"]):
                        continue
                    return mol_type

            # Priority 4: Check description keywords
            description = node_data.get("description", "").lower()
            for mol_type, indicators in type_indicators.items():
                keywords = indicators["description_keywords"]
                if any(kw in description for kw in keywords):
                    # Special check: "protein" in description but not "gene"
                    if mol_type == "protein" and "gene" in description:
                        continue
                    return mol_type

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

        # Extract captions for all molecule types from nodes
        captions = {"dna": None, "rna": None, "protein": None}
        caption_attached = False

        for node in nodes:
            _, node_data = node

            # Check for pre-extracted captions (from partition_service)
            for mol_type in ["dna", "rna", "protein"]:
                caption_key = f"{mol_type}_caption"
                if caption_key in node_data and node_data[caption_key]:
                    captions[mol_type] = node_data[caption_key]
                    caption_attached = True

            # If no pre-extracted captions, extract from node_data using the detected molecule_type
            if not caption_attached:
                caption = self._extract_caption(node_data, molecule_type)
                if caption:
                    captions[molecule_type] = caption
                    caption_attached = True
                    break  # Only need to extract once per batch

        # Attach all captions to QA pairs
        for qa in qa_pairs.values():
            qa["dna"] = captions["dna"] if captions["dna"] else ""
            qa["rna"] = captions["rna"] if captions["rna"] else ""
            qa["protein"] = captions["protein"] if captions["protein"] else ""

        if not caption_attached:
            node_sample = (
                dict(list(nodes[0][1].items())[:5]) if nodes else 'No nodes'
            )
            logger.warning(
                "No caption extracted for molecule_type=%s. Node data sample: %s",
                molecule_type, node_sample
            )

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
        # Extract QA pairs and molecule captions
        qa_items = [
            {
                "question": v["question"],
                "answer": v["answer"],
                "dna": v.get("dna", ""),
                "rna": v.get("rna", ""),
                "protein": v.get("protein", ""),
            }
            for item in results
            for k, v in item.items()
        ]

        # Format based on output format
        if output_data_format == "Alpaca":
            return [
                {
                    "instruction": qa["question"],
                    "input": "",
                    "output": qa["answer"],
                    "dna": qa["dna"],
                    "rna": qa["rna"],
                    "protein": qa["protein"],
                }
                for qa in qa_items
            ]
        if output_data_format == "Sharegpt":
            return [
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": [
                                {
                                    "text": qa["question"],
                                    "dna": qa["dna"],
                                    "rna": qa["rna"],
                                    "protein": qa["protein"],
                                }
                            ],
                        },
                        {"from": "gpt", "value": qa["answer"]},
                    ]
                }
                for qa in qa_items
            ]
        if output_data_format == "ChatML":
            return [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": qa["question"],
                                    "dna": qa["dna"],
                                    "rna": qa["rna"],
                                    "protein": qa["protein"],
                                }
                            ],
                        },
                        {"role": "assistant", "content": qa["answer"]},
                    ]
                }
                for qa in qa_items
            ]
        else:
            raise ValueError(f"Unknown output data format: {output_data_format}")
