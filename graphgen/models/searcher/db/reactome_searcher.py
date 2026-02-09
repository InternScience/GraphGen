import re
import time
from typing import Any, Dict, Optional

import requests
from requests.exceptions import RequestException

from graphgen.utils import logger


class ReactomeSearcher:
    """
    Reactome Pathway Search client for retrieving biological pathways by UniProt ID.

    Supports:
    1) Search pathways associated with a protein by UniProt accession.
    2) Rank pathways by relevance (curated vs inferred, diagram availability).
    3) Fetch detailed annotations for top-ranked pathways.

    API Documentation: https://reactome.org/ContentService
    """

    CONTENT_URL = "https://reactome.org/ContentService"

    # UniProt accession pattern (e.g., P04637, Q96KN2, O14763)
    UNIPROT_PATTERN = re.compile(
        r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
    )

    def __init__(
        self,
        timeout: int = 30,
        top_n_details: int = 5,
    ):
        """
        Initialize the Reactome Pathway Search client.

        Args:
            timeout: Request timeout in seconds.
            top_n_details: Number of top pathways to fetch detailed annotations for.
        """
        self.timeout = timeout
        self.top_n_details = top_n_details
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    @staticmethod
    def _is_uniprot_accession(text: str) -> bool:
        """Check if text is a valid UniProt accession number."""
        if not text or not isinstance(text, str):
            return False
        return bool(ReactomeSearcher.UNIPROT_PATTERN.match(text.strip()))

    def _calculate_relevance_score(self, pathway: Dict[str, Any]) -> int:
        """
        Calculate relevance score for pathway ranking.
        Higher score indicates higher biological significance.

        Scoring criteria:
        - Manual curation (not inferred): +10
        - Has pathway diagram: +5
        - Disease-related: +3
        - Specific biological terms in name: +2
        """

        # TODO: complete this function

        score = 0

        # Prioritize manually curated over computational predictions
        # Note: Mapping API may not return this, default to False
        if not pathway.get("isInferred", False):
            score += 10

        # Visual representations indicate well-characterized pathways
        # Note: Mapping API may not return this, default to False
        if pathway.get("hasDiagram", False):
            score += 5

        # Disease pathways often have higher clinical relevance
        # Note: Mapping API may not return this, default to False
        if pathway.get("isInDisease", False):
            score += 3

        # Prefer specific pathway types over generic classifications
        name = pathway.get("displayName", "").lower()
        specific_terms = [
            "signaling",
            "regulation",
            "activation",
            "pathway",
            "synthesis",
            "degradation",
            "repair",
            "apoptosis",
        ]
        if any(term in name for term in specific_terms):
            score += 2

        return score

    def _fetch_pathway_details(self, pathway_stid: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific pathway.

        Args:
            pathway_stid: Reactome stable ID (e.g., "R-HSA-111288").

        Returns:
            Dictionary with detailed annotations or None if fetch fails.
        """
        url = f"{self.CONTENT_URL}/data/query/{pathway_stid}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 404:
                logger.warning("Pathway %s not found in Reactome", pathway_stid)
                return None

            response.raise_for_status()
            data = response.json()

            # Extract key annotations
            details = {
                "schemaClass": data.get("schemaClass"),
                "summation": data.get("summation", [None])[0]
                if data.get("summation")
                else None,
                "compartment": [
                    c.get("displayName") for c in data.get("compartment", [])
                ],
                "disease": [d.get("displayName") for d in data.get("disease", [])],
                "sub_pathways": [
                    {"stId": e.get("stId"), "name": e.get("displayName")}
                    for e in data.get("hasEvent", [])[:5]  # First 5 sub-events
                ],
                "literature_references": [
                    {
                        "pubMedId": ref.get("pubMedIdentifier"),
                        "title": ref.get("displayName"),
                    }
                    for ref in data.get("literatureReference", [])[:3]  # Top 3 refs
                ],
            }

            return details

        except RequestException as e:
            logger.error("Failed to fetch details for pathway %s: %s", pathway_stid, e)
            return None

    def search_by_uniprot_id(self, accession: str) -> Optional[Dict]:
        """
        Search Reactome pathways by UniProt accession number.

        Retrieves all pathways associated with the protein using the dedicated
        mapping endpoint, ranks them by relevance, and fetches detailed
        annotations for the top N pathways.

        Args:
            accession: UniProt accession number (e.g., "P04637" for TP53).

        Returns:
            Dictionary with pathway information or None if search fails:
            {
                "molecule_type": "protein",
                "database": "Reactome",
                "id": accession,
                "content": {
                    "total_found": int,
                    "pathways": List[Dict]  # Top pathways with details
                },
                "url": str  # Link to Reactome search
            }
        """
        if not self._is_uniprot_accession(accession):
            logger.error("Invalid UniProt accession format: %s", accession)
            return None

        accession = accession.strip().upper()
        logger.debug("Searching Reactome pathways for %s", accession)

        # Step 1: Use the correct mapping endpoint for UniProt to pathways
        url = f"{self.CONTENT_URL}/data/mapping/UniProt/{accession}/pathways"
        params = {
            "interactors": "false",  # Exclude inferred from interactors for cleaner results
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)

            if response.status_code == 404:
                logger.info("No pathways found for %s", accession)
                return None

            response.raise_for_status()

            # The mapping API returns a list directly, not wrapped in searchHits
            pathways_data = response.json()

            if not pathways_data:
                logger.info("No pathways found for %s", accession)
                return None

            # Step 2: Use pathway data as-is
            pathways = []
            for pw in pathways_data:
                if isinstance(pw, dict):
                    pathways.append(pw)

            logger.info("Found %d pathways for %s", len(pathways), accession)

            # Step 3: Rank by relevance score
            # Note: Since mapping API doesn't return isInferred/hasDiagram/isInDisease,
            # we fetch details for pathways to get accurate scores if needed,
            # or use name-based heuristics. Here we rank by available info.
            scored = [(self._calculate_relevance_score(pw), pw) for pw in pathways]
            scored.sort(key=lambda x: x[0], reverse=True)
            sorted_pathways = [pw for _, pw in scored]

            # Step 4: Fetch details for top N pathways
            top_pathways = []
            for i, pw in enumerate(sorted_pathways[: self.top_n_details]):
                details = self._fetch_pathway_details(pw["stId"])
                if details:
                    pw["details"] = details
                    # Update scoring fields if details contain them
                    # (Details don't have these either, but keeping structure consistent)

                    # Small delay to avoid overwhelming API
                    if i < self.top_n_details - 1:
                        time.sleep(0.1)
                else:
                    pw["details"] = None

                top_pathways.append(pw)

            # Construct result in standard format
            result = {
                "molecule_type": "protein",
                "database": "Reactome",
                "id": accession,
                "content": {
                    "total_found": len(pathways),
                    "pathways": top_pathways,
                },
                "url": f"https://reactome.org/content/query?q={accession}",
            }

            return result

        except RequestException as e:
            logger.error("Failed to search Reactome for %s: %s", accession, e)
            return None

    def search(self, query: str, **kwargs) -> Optional[Dict]:
        """
        Search Reactome for pathway information.

        Args:
            query: Search query (UniProt accession number).
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with pathway information or None if not found.
        """
        if not query or not isinstance(query, str):
            logger.error("Empty or invalid input for Reactome search")
            return None

        query = query.strip()
        logger.debug("Reactome search query: %s", query)

        if self._is_uniprot_accession(query):
            logger.debug("Detected UniProt accession: %s", query)
            result = self.search_by_uniprot_id(query)
        else:
            raise ValueError(
                "ReactomeSearcher only supports UniProt accession numbers as queries."
            )

        if result:
            result["_search_query"] = query

        return result
