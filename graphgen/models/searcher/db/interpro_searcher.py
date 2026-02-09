import re
import time
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
    2) Search with protein sequence using EBI InterProScan API.
    3) Parse domain matches and associated GO terms, pathways.

    API Documentation: https://www.ebi.ac.uk/Tools/services/rest/iprscan5
    """

    def __init__(
        self,
        email: str = "graphgen@example.com",
        api_timeout: int = 30,
    ):
        """
        Initialize the InterPro Search client.

        Args:
            email (str): Email address for EBI API requests.
            api_timeout (int): Request timeout in seconds.
        """
        self.base_url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
        self.email = email
        self.api_timeout = api_timeout
        self.poll_interval = 5  # Fixed interval between status checks
        self.max_polls = 120  # Maximum polling attempts (10 minutes with 5s interval)

    @staticmethod
    def _is_protein_sequence(text: str) -> bool:
        """Check if text looks like a protein sequence."""
        # Remove common FASTA header prefix
        if text.startswith(">"):
            text = "\n".join(text.split("\n")[1:])
        # Check if contains mostly protein amino acids
        text = text.strip().replace("\n", "").replace(" ", "")
        # Protein sequences contain only A-Z letters (standard amino acids)
        return bool(re.fullmatch(r"[A-Z]+", text, re.I)) and len(text) > 10

    @staticmethod
    def _is_uniprot_accession(text: str) -> bool:
        """Check if text looks like a UniProt accession number."""
        # UniProt: 6-10 chars starting with letter, e.g., P01308, Q96KN2
        return bool(re.fullmatch(r"[A-Z][A-Z0-9]{5,9}", text.strip(), re.I))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def _submit_job(self, sequence: str, title: str = "") -> Optional[str]:
        """
        Submit a protein sequence for InterProScan analysis.

        Args:
            sequence (str): Protein sequence (FASTA or raw).
            title (str): Optional job title.

        Returns:
            Job ID if successful, None otherwise.
        """
        url = f"{self.base_url}/run"

        # Parse sequence if FASTA format
        if sequence.startswith(">"):
            sequence = (
                "\n".join(sequence.split("\n")[1:]).replace("\n", "").replace(" ", "")
            )

        params = {
            "email": self.email,
            "title": title or "GraphGen_Analysis",
            "sequence": sequence,
            "stype": "p",
            "appl": "NCBIfam,SMART,CDD,HAMAP",  # Multiple databases
            "goterms": "true",
            "pathways": "true",
            "format": "json",
        }

        try:
            response = requests.post(url, data=params, timeout=self.api_timeout)
            if response.status_code == 200:
                job_id = response.text.strip()
                logger.debug("InterProScan job submitted: %s", job_id)
                return job_id
            logger.error(
                "Failed to submit InterProScan job: %d - %s",
                response.status_code,
                response.text,
            )
            return None
        except RequestException as e:
            logger.error("Request error while submitting job: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def _check_status(self, job_id: str) -> Optional[str]:
        """Check the status of a submitted job."""
        url = f"{self.base_url}/status/{job_id}"
        try:
            response = requests.get(url, timeout=self.api_timeout)
            if response.status_code == 200:
                return response.text.strip()
            logger.warning(
                "Failed to check job status for %s: %d",
                job_id,
                response.status_code,
            )
            return None
        except RequestException as e:
            logger.error("Request error while checking status: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def _get_results(self, job_id: str) -> Optional[dict]:
        """Retrieve the analysis results for a completed job."""
        url = f"{self.base_url}/result/{job_id}/json"
        try:
            response = requests.get(url, timeout=self.api_timeout)
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "Failed to retrieve results for job %s: %d",
                job_id,
                response.status_code,
            )
            return None
        except RequestException as e:
            logger.error("Request error while retrieving results: %s", e)
            raise

    def _poll_job(self, job_id: str) -> Optional[dict]:
        """
        Poll a job until completion and retrieve results.

        Args:
            job_id (str): The job ID to poll.

        Returns:
            Results dictionary if successful, None otherwise.
        """
        for attempt in range(self.max_polls):
            status = self._check_status(job_id)

            if status == "FINISHED":
                logger.debug(
                    "Job %s completed after %d polls",
                    job_id,
                    attempt + 1,
                )
                return self._get_results(job_id)

            if status in ["FAILED", "NOT_FOUND"]:
                logger.warning("Job %s has status: %s", job_id, status)
                return None

            if status == "RUNNING":
                logger.debug(
                    "Job %s still running (attempt %d/%d)",
                    job_id,
                    attempt + 1,
                    self.max_polls,
                )
                time.sleep(self.poll_interval)
            else:
                logger.debug("Job %s status: %s", job_id, status)
                time.sleep(self.poll_interval)

        logger.warning(
            "Job %s polling timed out after %d attempts", job_id, self.max_polls
        )
        return None

    def search_by_sequence(self, sequence: str) -> Optional[Dict]:
        """
        Search for protein domains in a sequence using InterProScan API.

        Args:
            sequence (str): Protein sequence in FASTA or raw format.

        Returns:
            Dictionary with domain analysis results or None if failed.
        """
        if not sequence or not isinstance(sequence, str):
            logger.error("Invalid sequence provided")
            return None

        sequence = sequence.strip()

        if not self._is_protein_sequence(sequence):
            logger.error("Invalid protein sequence format")
            return None

        # Submit job
        job_id = self._submit_job(sequence)
        if not job_id:
            logger.error("Failed to submit InterProScan job")
            return None

        # Poll for results
        results = self._poll_job(job_id)
        if not results:
            logger.error("Failed to retrieve InterProScan results for job %s", job_id)
            return None

        return {
            "molecule_type": "protein",
            "database": "InterPro",
            "job_id": job_id,
            "content": results,
            "url": f"https://www.ebi.ac.uk/interpro/result/{job_id}/",
        }

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
        if not accession or not isinstance(accession, str):
            logger.error("Invalid accession provided")
            return None

        accession = accession.strip().upper()

        # Query InterPro REST API for UniProt entry
        url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{accession}/"

        response = requests.get(url, timeout=self.api_timeout)

        if response.status_code != 200:
            logger.warning(
                "Failed to search InterPro for accession %s: %d",
                accession,
                response.status_code,
            )
            return None

        data = response.json()

        result = {
            "molecule_type": "protein",
            "database": "InterPro",
            "id": accession,
            "content": data.get("results", []),
            "url": f"https://www.ebi.ac.uk/interpro/protein/uniprot/{accession}/",
        }

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def search(self, query: str, **kwargs) -> Optional[Dict]:
        """
        Search InterPro for protein domain information.

        Automatically detects query type:
        - UniProt accession number → lookup pre-computed domains
        - Protein sequence (FASTA or raw) → submit for InterProScan analysis

        Args:
            query (str): Search query (UniProt ID or protein sequence).
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with domain information or None if not found.
        """
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input")
            return None

        query = query.strip()
        logger.debug("InterPro search query: %s", query[:100])

        result = None

        # Check if UniProt accession
        if self._is_uniprot_accession(query):
            logger.debug("Detected UniProt accession: %s", query)
            result = self.search_by_uniprot_id(query)

        # Check if protein sequence
        elif self._is_protein_sequence(query):
            logger.debug("Detected protein sequence (length: %d)", len(query))
            result = self.search_by_sequence(query)

        else:
            # Try as UniProt ID first (in case format is non-standard)
            logger.debug("Trying as UniProt accession: %s", query)
            result = self.search_by_uniprot_id(query)

        if result:
            result["_search_query"] = query

        return result
