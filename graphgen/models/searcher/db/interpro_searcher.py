import re
from typing import Dict, Optional

from graphgen.bases import BaseSearcher
from graphgen.models.searcher.http_client import HTTPClient
from graphgen.utils import logger


class InterProSearch(BaseSearcher):
    """
    InterPro Search client to search protein domains and functional annotations.
    Supports:
    1) Get protein domain information by UniProt accession number.

    API Documentation: https://www.ebi.ac.uk/interpro/api/
    """

    def __init__(
        self,
        api_timeout: int = 30,
        qps: float = 10,
        max_concurrent: int = 5,
    ):
        """
        Initialize the InterPro Search client.

        Args:
            api_timeout (int): Request timeout in seconds.
        """
        self.http_client = HTTPClient(
            base_url="https://www.ebi.ac.uk/interpro/api",
            timeout=api_timeout,
            qps=qps,
            max_concurrent=max_concurrent,
            headers={
                "Accept": "application/json",
            },
        )

    @staticmethod
    def _is_uniprot_accession(text: str) -> bool:
        """Check if text looks like a UniProt accession number."""
        return bool(re.fullmatch(r"[A-Z][A-Z0-9]{5,9}", text.strip(), re.I))

    async def search_by_uniprot_id(self, accession: str) -> Optional[Dict]:
        """
        Search InterPro database by UniProt accession number.

        Args:
            accession (str): UniProt accession number.

        Returns:
            Dictionary with domain information or None if not found.
        """
        if (
            not accession
            or not isinstance(accession, str)
            or not self._is_uniprot_accession(accession)
        ):
            logger.error("Invalid accession provided")
            return None

        accession = accession.strip().upper()
        endpoint = f"entry/interpro/protein/uniprot/{accession}/"

        try:
            data = await self.http_client.aget(endpoint)
        except Exception as e:
            logger.warning(
                "Failed to search InterPro for accession %s: %s",
                accession,
                str(e),
            )
            return None

        for result in data.get("results", []):
            interpro_acc = result.get("metadata", {}).get("accession")
            if interpro_acc:
                entry_details = await self.get_entry_details(interpro_acc)
                if entry_details:
                    result["entry_details"] = entry_details

        return {
            "molecule_type": "protein",
            "database": "InterPro",
            "id": accession,
            "content": data.get("results", []),
            "url": f"https://www.ebi.ac.uk/interpro/protein/uniprot/{accession}/",
        }

    async def get_entry_details(self, interpro_accession: str) -> Optional[Dict]:
        """
        Get detailed information for a specific InterPro entry.

        Args:
            interpro_accession (str): InterPro accession number (e.g., IPR000001).
        Returns:
            Dictionary with entry details or None if not found.
        """
        if not interpro_accession or not isinstance(interpro_accession, str):
            return None

        endpoint = f"entry/interpro/{interpro_accession}/"
        try:
            return await self.http_client.aget(endpoint)
        except Exception as e:
            logger.warning(
                "Failed to get InterPro entry %s: %s",
                interpro_accession,
                str(e),
            )
            return None

    async def search(self, query: str, **kwargs) -> Optional[Dict]:
        """
        Search InterPro for protein domain information by UniProt accession.

        Args:
            query (str): UniProt accession number (e.g., P01308, Q96KN2).
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with domain information or None if not found.
        """
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input")
            return None

        query = query.strip()

        logger.debug("InterPro search query: %s", query[:100])
        result = await self.search_by_uniprot_id(query)
        logger.debug("InterPro search result: %s", str(result)[:100])

        if result:
            result["_search_query"] = query

        return result
