import io
import re
import time
from typing import Dict, List, Optional, Union

import requests
from Bio import SeqIO
from requests.exceptions import RequestException, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher
from graphgen.utils import logger


class RfamSearcher(BaseSearcher):
    """
    Rfam Search client to query RNA families and non-coding RNA databases.

    Rfam is a collection of non-coding RNA (ncRNA) families represented by
    multiple sequence alignments and consensus secondary structures. Each family
    is described by a covariance model (CM) used for identifying new members.

    Features:
    1) Get RNA family by accession (RFXXXXX) or ID (e.g., snoZ107_R87)
    2) Search families by keywords using EBI search
    3) Search sequences against Rfam covariance models (via RNAcentral integration)
    4) Retrieve covariance models, alignments, and secondary structures

    API Documentation: https://docs.rfam.org/en/latest/api.html
    """

    BASE_URL = "https://rfam.org/family"
    EBI_SEARCH_URL = "https://www.ebi.ac.uk/ebisearch/ws/rest/rfam"
    RNACENTRAL_SEARCH_URL = "https://rnacentral.org/api/v1/sequence-search"

    # Supported return formats
    FORMATS = ["json", "xml", "text"]

    # RNA molecule type mapping
    RNA_TYPES = {
        "Gene": "gene",
        "snRNA": "snrna",
        "snoRNA": "snorna",
        "rRNA": "rrna",
        "tRNA": "trna",
        "miRNA": "mirna",
        "ribozyme": "ribozyme",
        "riboswitch": "riboswitch",
    }

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        poll_interval: int = 5,
    ):
        """
        Initialize Rfam searcher.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            poll_interval: Seconds between polling for async sequence search results
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def _is_valid_accession(self, query: str) -> bool:
        """Check if query is a valid Rfam accession (RF + 5 digits)."""
        return bool(re.fullmatch(r"RF\d{5}", query, re.IGNORECASE))

    def _is_valid_id(self, query: str) -> bool:
        """Check if query looks like a valid Rfam ID (alphanumeric with underscores)."""
        return bool(re.fullmatch(r"[a-zA-Z0-9_]+", query)) and not query.startswith(
            "RF"
        )

    def _get(
        self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None
    ) -> Optional[Union[Dict, str]]:
        """
        Perform GET request with retry logic.

        Args:
            url: Full URL to request
            params: Query parameters
            headers: Optional headers override

        Returns:
            Parsed response (JSON dict or text) or None if not found
        """
        request_headers = headers or self.session.headers

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((RequestException, Timeout)),
            reraise=True,
        )
        def _request():
            response = self.session.get(
                url, params=params, headers=request_headers, timeout=self.timeout
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text

        try:
            return _request()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("Rfam resource not found: %s", url)
                return None
            raise
        except Exception as exc:
            logger.error("Request failed for %s: %s", url, exc)
            raise

    def get_by_accession(self, accession: str, format: str = "json") -> Optional[Dict]:
        """
        Retrieve detailed information about an Rfam family by accession or ID.

        Args:
            accession: Rfam accession (e.g., RF00360) or family ID (e.g., snoZ107_R87)
            format: Response format - 'json', 'xml', or 'text'

        Returns:
            Dictionary containing family information or None if not found
        """
        if not accession or not isinstance(accession, str):
            logger.error("Invalid accession provided")
            return None

        accession = accession.strip()
        format = format.lower()
        if format not in self.FORMATS:
            format = "json"

        logger.debug("Fetching Rfam family: %s", accession)

        # Build URL with content type
        if format == "json":
            url = f"{self.BASE_URL}/{accession}?content-type=application/json"
        elif format == "xml":
            url = f"{self.BASE_URL}/{accession}?content-type=text/xml"
        else:
            url = f"{self.BASE_URL}/{accession}"

        result = self._get(url)
        if not result:
            return None

        if format == "json" and isinstance(result, dict):
            return self._normalize_family(result)
        # For XML/text, wrap in dict with raw content
        return {
            "molecule_type": "rna_family",
            "database": "Rfam",
            "id": accession,
            "raw_content": result,
            "url": f"https://rfam.org/family/{accession}",
        }

    def _normalize_family(self, data: Dict) -> Dict:
        """
        Normalize Rfam family data to standard format.

        Args:
            data: Raw API response (contains 'rfam' key)

        Returns:
            Standardized dictionary
        """
        rfam_data = data.get("rfam", data)  # Handle both wrapped and unwrapped

        entry = rfam_data.get("entry", {})
        curation = rfam_data.get("curation_details", {})
        cm_details = rfam_data.get("cm_details", {})
        cutoffs = cm_details.get("cutoffs", {})

        # Extract RNA type from curation details
        rna_type = "ncRNA"
        type_str = curation.get("type", "")
        if type_str:
            # Parse type string like "Gene; snRNA; snoRNA; CD-box;"
            types = [t.strip() for t in type_str.split(";") if t.strip()]
            if types:
                rna_type = (
                    types[-1]
                    if types[-1] != "CD-box"
                    else types[-2]
                    if len(types) > 1
                    else types[0]
                )

        normalized = {
            "molecule_type": "rna_family",
            "database": "Rfam",
            "id": entry.get("accession", "Unknown"),
            "family_id": entry.get("id", "Unknown"),
            "name": entry.get("id", "Unknown"),  # Human readable ID
            "description": entry.get("description", ""),
            "comment": entry.get("comment", ""),
            "rna_type": rna_type,
            "url": f"https://rfam.org/family/{entry.get('accession', '')}",
            # Curation details
            "author": curation.get("author", ""),
            "seed_source": curation.get("seed_source", ""),
            "structure_source": curation.get("structure_source", ""),
            "num_sequences": {
                "seed": curation.get("num_seqs", {}).get("seed", 0),
                "full": curation.get("num_seqs", {}).get("full", 0),
            },
            "num_species": curation.get("num_species", 0),
            # Covariance model details
            "cutoffs": {
                "gathering": float(cutoffs.get("gathering", 0)),
                "trusted": float(cutoffs.get("trusted", 0)),
                "noise": float(cutoffs.get("noise", 0)),
            },
            "cm_commands": {
                "build": cm_details.get("build_command", ""),
                "search": cm_details.get("search_command", ""),
            },
        }

        return normalized

    def accession_to_id(self, accession: str) -> Optional[str]:
        """
        Convert Rfam accession to family ID.

        Args:
            accession: Rfam accession (e.g., RF00360)

        Returns:
            Family ID (e.g., snoZ107_R87) or None
        """
        if not self._is_valid_accession(accession):
            logger.error("Invalid Rfam accession format: %s", accession)
            return None

        url = f"{self.BASE_URL}/{accession}/id"
        result = self._get(url, headers={"Accept": "text/plain"})

        if isinstance(result, str):
            return result.strip()

        return None

    def id_to_accession(self, family_id: str) -> Optional[str]:
        """
        Convert Rfam family ID to accession.

        Args:
            family_id: Family ID (e.g., snoZ107_R87)

        Returns:
            Accession (e.g., RF00360) or None
        """
        url = f"{self.BASE_URL}/{family_id}/acc"
        result = self._get(url, headers={"Accept": "text/plain"})

        if isinstance(result, str):
            return result.strip()
        return None

    def get_best_hit(self, keyword: str) -> Optional[Dict]:
        """
        Search Rfam families by keyword using EBI search and return the best hit.

        Args:
            keyword: Search term (e.g., 'riboswitch', 'tRNA', 'snoRNA')

        Returns:
            Best matching family or None
        """
        if not keyword or not isinstance(keyword, str):
            return None

        keyword = keyword.strip()
        if not keyword:
            return None

        logger.debug("Searching Rfam for keyword: %s", keyword)

        # Use EBI search API
        params = {
            "query": keyword,
            "format": "json",
            "size": 1,  # Only get best hit
            "fields": "accession,description,num_seed,id",
        }

        try:
            result = self._get(self.EBI_SEARCH_URL, params=params)
            if not result or not result.get("entries"):
                logger.info("No Rfam results found for keyword: %s", keyword)
                return None

            # Get first entry
            entry = result["entries"][0]
            accession = entry.get("fields", {}).get("accession", [None])[0]

            if accession:
                return self.get_by_accession(accession)
            return None

        except Exception as exc:
            logger.error("Keyword search failed: %s", exc)
            return None

    def search_families(
        self, keyword: str, limit: int = 10, fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search Rfam families by keyword and return multiple results.

        Args:
            keyword: Search term
            limit: Maximum number of results
            fields: List of fields to retrieve (default: accession, description, id)

        Returns:
            List of family dictionaries
        """
        if not keyword:
            return []

        if fields is None:
            fields = ["accession", "description", "id", "num_seed"]

        params = {
            "query": keyword,
            "format": "json",
            "size": limit,
            "fields": ",".join(fields),
        }

        try:
            result = self._get(self.EBI_SEARCH_URL, params=params)
            if not result or not result.get("entries"):
                return []

            families = []
            for entry in result["entries"]:
                accession = entry.get("fields", {}).get("accession", [None])[0]
                if accession:
                    family = self.get_by_accession(accession)
                    if family:
                        families.append(family)

            return families

        except Exception as exc:
            logger.error("Family search failed: %s", exc)
            return []

    def search_by_sequence(
        self, sequence: str, timeout: int = 300, get_results: bool = True
    ) -> Optional[Dict]:
        """
        Search a nucleotide sequence against Rfam covariance models.

        Note: This uses the RNAcentral sequence search API which integrates Rfam.
        For large-scale analyses, consider using Infernal cmscan locally.

        Args:
            sequence: RNA/DNA sequence (raw or FASTA format)
            timeout: Maximum time to wait for results (seconds)
            get_results: If True, poll until results ready; if False, return job ID only

        Returns:
            Dictionary containing search results or job status
        """
        # Parse sequence
        seq = self._parse_sequence(sequence)
        if not seq:
            logger.error("Invalid sequence provided")
            return None

        # Validate sequence content (RNA/DNA)
        if not re.fullmatch(r"[ACGTURYSWKMBDHVN\s]+", seq, re.IGNORECASE):
            logger.error("Sequence contains invalid characters for RNA/DNA")
            return None

        logger.debug("Submitting sequence search to RNAcentral (length: %d)", len(seq))

        # Submit to RNAcentral sequence search (includes Rfam)
        payload = {
            "sequence": seq,
            "databases": ["rfam"],  # Focus on Rfam only
        }

        try:
            response = self.session.post(
                self.RNACENTRAL_SEARCH_URL,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            job_data = response.json()

            job_id = job_data.get("job_id")
            if not job_id:
                logger.error("No job ID returned from sequence search")
                return None

            if not get_results:
                return {
                    "job_id": job_id,
                    "status": "submitted",
                    "poll_url": f"{self.RNACENTRAL_SEARCH_URL}/{job_id}",
                }

            # Poll for results
            return self._poll_sequence_results(job_id, timeout)

        except Exception as exc:
            logger.error("Sequence search submission failed: %s", exc)
            return None

    def _poll_sequence_results(self, job_id: str, timeout: int) -> Optional[Dict]:
        """
        Poll RNAcentral for sequence search results.

        Args:
            job_id: Job ID from submission
            timeout: Maximum time to wait

        Returns:
            Search results dictionary
        """
        start_time = time.time()
        poll_url = f"{self.RNACENTRAL_SEARCH_URL}/{job_id}"

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(poll_url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                status = data.get("status", "unknown")

                if status == "completed":
                    return self._normalize_sequence_results(data)
                if status in ["failed", "error"]:
                    logger.error(
                        "Sequence search job failed: %s",
                        data.get("message", "Unknown error"),
                    )
                    return {
                        "status": "failed",
                        "error": data.get("message", "Unknown error"),
                    }

                # Still running, wait
                time.sleep(self.poll_interval)

            except Exception as exc:
                logger.error("Polling error: %s", exc)
                time.sleep(self.poll_interval)

        logger.error("Sequence search timed out after %d seconds", timeout)
        return {"status": "timeout", "job_id": job_id}

    def _normalize_sequence_results(self, data: Dict) -> Dict:
        """
        Normalize sequence search results.

        Args:
            data: Raw RNAcentral API response

        Returns:
            Standardized results dictionary
        """
        hits = data.get("results", [])

        normalized = {
            "molecule_type": "sequence_search",
            "database": "Rfam",
            "status": "completed",
            "total_hits": len(hits),
            "query_sequence": data.get("query_sequence", ""),
            "hits": [],
        }

        for hit in hits:
            rfam_data = hit.get("rfam", {})
            if not rfam_data:
                continue

            hit_info = {
                "family_accession": rfam_data.get("accession"),
                "family_id": rfam_data.get("id"),
                "family_name": rfam_data.get("description", ""),
                "bit_score": hit.get("score"),
                "e_value": hit.get("e_value"),
                "alignment": {
                    "query_start": hit.get("alignment", {}).get("query_start"),
                    "query_end": hit.get("alignment", {}).get("query_end"),
                    "subject_start": hit.get("alignment", {}).get("subject_start"),
                    "subject_end": hit.get("alignment", {}).get("subject_end"),
                    "strand": hit.get("alignment", {}).get("strand"),
                },
                "url": f"https://rfam.org/family/{rfam_data.get('accession', '')}",
            }
            normalized["hits"].append(hit_info)

        # Sort by E-value
        normalized["hits"].sort(key=lambda x: x.get("e_value", float("inf")))

        return normalized

    def _parse_sequence(self, sequence: str) -> Optional[str]:
        """
        Parse and clean input sequence.

        Args:
            sequence: Raw sequence or FASTA format

        Returns:
            Cleaned sequence string or None
        """
        try:
            if sequence.startswith(">"):
                # FASTA format
                with io.StringIO(sequence) as handle:
                    record = next(SeqIO.parse(handle, "fasta"))
                    return str(record.seq)
            else:
                # Raw sequence, remove whitespace
                return "".join(sequence.split())
        except Exception as exc:
            logger.error("Failed to parse sequence: %s", exc)
            return None

    def get_covariance_model(self, accession: str) -> Optional[str]:
        """
        Retrieve the covariance model (CM) for a family.

        Args:
            accession: Rfam accession

        Returns:
            CM file content as string or None
        """
        if not self._is_valid_accession(accession):
            logger.error("Invalid accession: %s", accession)
            return None

        url = f"{self.BASE_URL}/{accession}/cm"
        result = self._get(url, headers={"Accept": "text/plain"})

        if isinstance(result, str):
            return result
        return None

    def get_seed_alignment(
        self, accession: str, format: str = "stockholm"
    ) -> Optional[str]:
        """
        Retrieve the seed alignment for a family.

        Args:
            accession: Rfam accession
            format: 'stockholm' or 'fasta'

        Returns:
            Alignment content as string or None
        """
        if not self._is_valid_accession(accession):
            return None

        if format.lower() == "fasta":
            url = f"{self.BASE_URL}/{accession}/alignment?format=fasta"
        else:
            url = f"{self.BASE_URL}/{accession}/alignment?format=stockholm"

        result = self._get(url, headers={"Accept": "text/plain"})

        if isinstance(result, str):
            return result
        return None

    def get_secondary_structure_url(
        self, accession: str, image_type: str = "norm"
    ) -> str:
        """
        Get URL for secondary structure image.

        Args:
            accession: Rfam accession
            image_type: One of 'norm', 'cons', 'cov', 'fchp', 'ent', 'maxcm', 'rscape', 'rscape-cyk'

        Returns:
            URL string
        """
        valid_types = [
            "norm",
            "cons",
            "cov",
            "fchp",
            "ent",
            "maxcm",
            "rscape",
            "rscape-cyk",
        ]
        if image_type not in valid_types:
            image_type = "norm"

        return f"{self.BASE_URL}/{accession}/image/{image_type}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def search(self, query: str, **kwargs) -> Optional[Dict]:
        """
        Unified search interface for Rfam.

        Auto-detects query type:
        - Rfam accession (RFXXXXX): Direct family lookup
        - Rfam ID (text): Family lookup by ID
        - FASTA sequence: Sequence search against Rfam
        - Single word: Keyword search

        Args:
            query: Search query
            **kwargs: Additional parameters:
                - format: 'json' or 'xml' for family lookups
                - limit: Max results for keyword search
                - timeout: Timeout for sequence search

        Returns:
            Dictionary containing search results
        """
        if not query or not isinstance(query, str):
            logger.error("Empty or invalid query")
            return None

        query = query.strip()
        if not query:
            return None

        logger.debug("Rfam search query: %s", query)

        result = None

        # Check if it's an accession (RF + 5 digits)
        if self._is_valid_accession(query):
            format_type = kwargs.get("format", "json")
            result = self.get_by_accession(query, format=format_type)
        # Check if it looks like a family ID (alphanumeric with underscores, not RF prefix)
        elif self._is_valid_id(query):
            # Try to get by ID first
            result = self.get_by_accession(query)
        # Check if FASTA format
        elif query.startswith(">"):
            timeout = kwargs.get("timeout", 300)
            result = self.search_by_sequence(query, timeout=timeout)
        # Check if raw sequence (long string of nucleotide chars)
        elif len(query) > 20 and re.fullmatch(
            r"[ACGTURYSWKMBDHVN\s]+", query, re.IGNORECASE
        ):
            timeout = kwargs.get("timeout", 300)
            result = self.search_by_sequence(query, timeout=timeout)
        else:
            # Treat as keyword
            limit = kwargs.get("limit", 1)
            if limit == 1:
                result = self.get_best_hit(query)
            else:
                families = self.search_families(query, limit=limit)
                result = {
                    "molecule_type": "search_results",
                    "database": "Rfam",
                    "query": query,
                    "count": len(families),
                    "families": families,
                }

        if result and isinstance(result, dict):
            result["_search_query"] = query
        return result

    def __del__(self):
        """Cleanup session."""
        if hasattr(self, "session"):
            self.session.close()
