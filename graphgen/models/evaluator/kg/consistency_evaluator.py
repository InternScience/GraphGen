import asyncio
import json
import re
from typing import Any, Dict, List

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.utils import create_event_loop, logger


class ConsistencyEvaluator:
    """Evaluates consistency by detecting semantic conflicts using LLM-as-a-Judge.

    For entities with multiple source chunks, compares entity_type and description
    extracted from different chunks to detect semantic conflicts.
    """

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        chunk_storage: BaseKVStorage,
        llm_client: BaseLLMWrapper,
        max_concurrent: int = 10,
    ):
        self.graph_storage = graph_storage
        self.chunk_storage = chunk_storage
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate consistency by detecting semantic conflicts."""
        all_nodes = self.graph_storage.get_all_nodes() or []
        if not all_nodes:
            return {"error": "Empty graph"}

        loop = create_event_loop()
        return loop.run_until_complete(self._evaluate_consistency(all_nodes))

    async def _evaluate_consistency(self, all_nodes: List) -> Dict[str, Any]:
        """Async evaluation of consistency."""
        # Filter entities with multiple source chunks
        entities_with_multiple_sources = []
        for node_id, node_data in all_nodes:
            if not isinstance(node_data, dict):
                continue
            source_ids = node_data.get("source_id", "").split("<SEP>")
            source_ids = [sid.strip() for sid in source_ids if sid.strip()]
            if len(source_ids) > 1:  # Only check entities from multiple chunks
                entities_with_multiple_sources.append((node_id, node_data, source_ids))

        if not entities_with_multiple_sources:
            logger.info(
                "No entities with multiple sources found, skipping consistency check"
            )
            return {
                "conflict_rate": 0.0,
                "conflict_entities_count": 0,
                "total_entities": len(all_nodes),
                "conflicts": [],
            }

        logger.info(
            f"Checking consistency for {len(entities_with_multiple_sources)} entities with multiple sources"
        )

        # Evaluate entities concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_entity(entity_info):
            async with semaphore:
                return await self._evaluate_entity_consistency(entity_info)

        tasks = [
            evaluate_entity(entity_info)
            for entity_info in entities_with_multiple_sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        conflicts = []
        conflict_entities = set()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to evaluate entity {entities_with_multiple_sources[i][0]}: {result}"
                )
                continue

            entity_id, entity_conflicts = result
            if entity_conflicts:
                conflicts.extend(entity_conflicts)
                conflict_entities.add(entity_id)

        total_entities = len(all_nodes)
        conflict_rate = (
            len(conflict_entities) / total_entities if total_entities > 0 else 0
        )

        return {
            "conflict_rate": conflict_rate,
            "conflict_entities_count": len(conflict_entities),
            "total_entities": total_entities,
            "entities_checked": len(entities_with_multiple_sources),
            "conflicts": conflicts[:100],  # Limit to first 100 conflicts
        }

    def _clean_entity_id(self, entity_id: str) -> str:
        """Clean entity ID by removing surrounding quotes."""
        clean_id = entity_id.strip()
        if (clean_id.startswith('"') and clean_id.endswith('"')) or (
            clean_id.startswith("'") and clean_id.endswith("'")
        ):
            clean_id = clean_id[1:-1].strip()
        return clean_id

    async def _evaluate_entity_consistency(
        self, entity_info: tuple
    ) -> tuple[str, List[Dict]]:
        """Evaluate consistency for a single entity."""
        entity_id, _node_data, source_ids = entity_info
        # Clean entity_id for display
        clean_entity_id = self._clean_entity_id(entity_id)
        conflicts = []

        # Get chunks for this entity
        chunks = self._get_entity_chunks(source_ids)
        if len(chunks) < 2:
            return entity_id, []

        # Extract entity attributes from each chunk
        entity_extractions = {}
        for chunk in chunks:
            extraction = await self._extract_entity_from_chunk(entity_id, chunk)
            if extraction:
                entity_extractions[chunk.id] = extraction

        if len(entity_extractions) < 2:
            return entity_id, []

        # Check entity type consistency
        type_extractions = {
            chunk_id: ext.get("entity_type", "")
            for chunk_id, ext in entity_extractions.items()
        }
        type_conflict = await self._check_entity_type_consistency(
            entity_id, type_extractions
        )
        if type_conflict and type_conflict.get("has_conflict", False):
            conflicts.append(
                {
                    "entity_id": clean_entity_id,
                    "conflict_type": "entity_type",
                    "conflict_severity": type_conflict.get("conflict_severity", 0.0),
                    "conflict_reasoning": type_conflict.get("conflict_reasoning", ""),
                    "conflicting_values": type_conflict.get("conflicting_types", []),
                    "recommended_value": type_conflict.get("recommended_type", ""),
                }
            )

        # Check entity description consistency
        descriptions = {
            chunk_id: ext.get("description", "")
            for chunk_id, ext in entity_extractions.items()
        }
        desc_conflict = await self._check_entity_description_consistency(
            entity_id, descriptions
        )
        if desc_conflict and desc_conflict.get("has_conflict", False):
            conflicts.append(
                {
                    "entity_id": clean_entity_id,
                    "conflict_type": "description",
                    "conflict_severity": desc_conflict.get("conflict_severity", 0.0),
                    "conflict_reasoning": desc_conflict.get("conflict_reasoning", ""),
                    "conflicting_values": desc_conflict.get(
                        "conflicting_descriptions", []
                    ),
                    "conflict_details": desc_conflict.get("conflict_details", ""),
                }
            )

        return entity_id, conflicts

    def _get_entity_chunks(self, source_ids: List[str]) -> List[Chunk]:
        """Get all chunks related to an entity."""
        chunks = []
        for chunk_id in source_ids:
            chunk_data = self.chunk_storage.get_by_id(chunk_id)
            if chunk_data:
                try:
                    chunk = Chunk.from_dict(chunk_id, chunk_data)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to load chunk {chunk_id}: {e}")
                    continue
        return chunks

    async def _extract_entity_from_chunk(
        self, entity_id: str, chunk: Chunk
    ) -> Dict[str, str]:
        """Extract entity attributes from a chunk using LLM."""
        try:
            # Clean entity_id: remove surrounding quotes if present
            clean_entity_id = self._clean_entity_id(entity_id)

            prompt = ENTITY_EXTRACTION_PROMPT.format(
                entity_name=clean_entity_id,
                chunk_content=chunk.content[:2000]
                if chunk.content
                else "",  # Limit content length
            )

            response = await self.llm_client.generate_answer(prompt)

            # Try to parse JSON response
            try:
                extraction = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    extraction = json.loads(json_match.group(0))
                else:
                    logger.warning(
                        f"Failed to parse extraction response for {entity_id} in chunk {chunk.id}"
                    )
                    return {}

            # Normalize entity_type to lowercase and validate
            entity_type = extraction.get("entity_type", "").lower().strip()
            # Valid preset types
            valid_types = {
                "concept",
                "date",
                "location",
                "keyword",
                "organization",
                "person",
                "event",
                "work",
                "nature",
                "artificial",
                "science",
                "technology",
                "mission",
                "gene",
            }
            # If entity_type is not in valid types, default to "concept"
            if entity_type not in valid_types:
                if entity_type:  # If LLM provided a type but it's invalid
                    logger.warning(
                        f"Invalid entity_type '{entity_type}' for entity {clean_entity_id} in chunk {chunk.id}, "
                        f"defaulting to 'concept'"
                    )
                entity_type = "concept"

            return {
                "entity_type": entity_type,
                "description": extraction.get("description", ""),
            }
        except Exception as e:
            logger.error(
                f"Error extracting entity {entity_id} from chunk {chunk.id}: {e}"
            )
            return {}

    async def _check_entity_type_consistency(
        self, entity_id: str, type_extractions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Check entity type consistency using LLM."""
        if len(set(type_extractions.values())) <= 1:
            # All types are the same, no conflict
            return {"has_conflict": False}

        try:
            type_list = [
                f"Chunk {chunk_id}: {entity_type}"
                for chunk_id, entity_type in type_extractions.items()
                if entity_type
            ]

            prompt = ENTITY_TYPE_CONFLICT_PROMPT.format(
                entity_name=entity_id, type_extractions="\n".join(type_list)
            )

            response = await self.llm_client.generate_answer(prompt)

            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    logger.warning(
                        f"Failed to parse conflict detection response for {entity_id}"
                    )
                    return {"has_conflict": False}

            return result
        except Exception as e:
            logger.error(f"Error checking type consistency for {entity_id}: {e}")
            return {"has_conflict": False}

    async def _check_entity_description_consistency(
        self, entity_id: str, descriptions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Check entity description consistency using LLM."""
        # Filter out empty descriptions
        valid_descriptions = {k: v for k, v in descriptions.items() if v}
        if len(valid_descriptions) < 2:
            return {"has_conflict": False}

        if len(set(valid_descriptions.values())) <= 1:
            # All descriptions are the same, no conflict
            return {"has_conflict": False}

        try:
            desc_list = [
                f"Chunk {chunk_id}: {description}"
                for chunk_id, description in valid_descriptions.items()
            ]

            prompt = ENTITY_DESCRIPTION_CONFLICT_PROMPT.format(
                entity_name=entity_id, descriptions="\n".join(desc_list)
            )

            response = await self.llm_client.generate_answer(prompt)

            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    logger.warning(
                        f"Failed to parse conflict detection response for {entity_id}"
                    )
                    return {"has_conflict": False}

            return result
        except Exception as e:
            logger.error(f"Error checking description consistency for {entity_id}: {e}")
            return {"has_conflict": False}

    async def _check_relation_consistency(
        self, src_id: str, dst_id: str, relation_extractions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Check relation consistency using LLM."""
        if len(set(relation_extractions.values())) <= 1:
            return {"has_conflict": False}

        try:
            rel_list = [
                f"Chunk {chunk_id}: {relation}"
                for chunk_id, relation in relation_extractions.items()
                if relation
            ]

            prompt = RELATION_CONFLICT_PROMPT.format(
                source_entity=src_id,
                target_entity=dst_id,
                relation_descriptions="\n".join(rel_list),
            )

            response = await self.llm_client.generate_answer(prompt)

            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    logger.warning(
                        f"Failed to parse relation conflict response for {src_id}->{dst_id}"
                    )
                    return {"has_conflict": False}

            return result
        except Exception as e:
            logger.error(
                f"Error checking relation consistency for {src_id}->{dst_id}: {e}"
            )
            return {"has_conflict": False}
