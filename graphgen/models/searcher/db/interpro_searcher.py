import re
from typing import Dict, Optional

import requests
from requests.exceptions import RequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher
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
    ):
        """
        Initialize the InterPro Search client.

        Args:
            api_timeout (int): Request timeout in seconds.
        """
        self.api_timeout = api_timeout
        self.BASE_URL = "https://www.ebi.ac.uk/interpro/api"

    @staticmethod
    def _is_uniprot_accession(text: str) -> bool:
        """Check if text looks like a UniProt accession number."""
        # UniProt: 6-10 chars starting with letter, e.g., P01308, Q96KN2
        return bool(re.fullmatch(r"[A-Z][A-Z0-9]{5,9}", text.strip(), re.I))

    def search_by_uniprot_id(self, accession: str) -> Optional[Dict]:
        """
        Search InterPro database by UniProt accession number.

        This method queries the EBI API to get pre-computed domain information
        for a known UniProt entry.

        Args:
            accession (str): UniProt accession number.

        Returns:
            Dictionary with domain information or None if not found.
        """
        if not accession or not isinstance(accession, str) or not self._is_uniprot_accession(accession):
            logger.error("Invalid accession provided")
            return None

        accession = accession.strip().upper()

        # Query InterPro REST API for UniProt entry
        url = f"{self.BASE_URL}/entry/interpro/protein/uniprot/{accession}/"

        response = requests.get(url, timeout=self.api_timeout)

        if response.status_code != 200:
            logger.warning(
                "Failed to search InterPro for accession %s: %d",
                accession,
                response.status_code,
            )
            return None

        data = response.json()

        # Get entry details for each InterPro entry found
        for result in data.get("results", []):
            interpro_acc = result.get("metadata", {}).get("accession")
            if interpro_acc:
                entry_details = self.get_entry_details(interpro_acc)
                if entry_details:
                    result["entry_details"] = entry_details

        result = {
            "molecule_type": "protein",
            "database": "InterPro",
            "id": accession,
            "content": data.get("results", []),
            "url": f"https://www.ebi.ac.uk/interpro/protein/uniprot/{accession}/",
        }

        return result

    def get_entry_details(self, interpro_accession: str) -> Optional[Dict]:
        """
        Get detailed information for a specific InterPro entry.

        Args:
            interpro_accession (str): InterPro accession number (e.g., IPR000001).
        Returns:
            Dictionary with entry details or None if not found.
        """
        if not interpro_accession or not isinstance(interpro_accession, str):
            return None

        url = f"{self.BASE_URL}/entry/interpro/{interpro_accession}/"

        response = requests.get(url, timeout=self.api_timeout)
        if response.status_code != 200:
            logger.warning(
                "Failed to get InterPro entry %s: %d",
                interpro_accession,
                response.status_code,
            )
            return None

        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def search(self, query: str, **kwargs) -> Optional[Dict]:
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

        # Search by UniProt ID
        logger.debug("Searching for UniProt accession: %s", query)
        result = self.search_by_uniprot_id(query)

        if result:
            result["_search_query"] = query

        return result
