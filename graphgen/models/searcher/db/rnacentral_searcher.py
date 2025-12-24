import asyncio
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import tempfile
from typing import Dict, Optional, List, Any, Set

import hashlib
import requests
import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher


@lru_cache(maxsize=None)
def _get_pool():
    return ThreadPoolExecutor(max_workers=20)  # NOTE：can increase for better parallelism

class RNACentralSearch(BaseSearcher):
    """
    RNAcentral Search client to search RNA databases.
    1) Get RNA by RNAcentral ID.
    2) Search with keywords or RNA names (fuzzy search).
    3) Search with RNA sequence.

    API Documentation: https://rnacentral.org/api/v1
    """

    def __init__(
        self,
        use_local_blast: bool = False,
        local_blast_db: str = "rna_db",
        api_timeout: int = 30,
        blast_num_threads: int = 4,
        working_dir: str = "cache",
    ):
        super().__init__(working_dir=working_dir)
        self.base_url = "https://rnacentral.org/api/v1"
        self.headers = {"Accept": "application/json"}
        self.use_local_blast = use_local_blast
        self.local_blast_db = local_blast_db
        self.api_timeout = api_timeout
        self.blast_num_threads = blast_num_threads  # Number of threads for BLAST search

        if self.use_local_blast and not os.path.isfile(f"{self.local_blast_db}.nhr"):
            self.logger.error("Local BLAST database files not found. Please check the path.")
            self.use_local_blast = False

    @staticmethod
    def _rna_data_to_dict(
        rna_id: str,
        rna_data: Dict[str, Any],
        xrefs_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        organisms, gene_names, so_terms = set(), set(), set()
        modifications: List[Any] = []

        for xref in xrefs_data or []:
            acc = xref.get("accession", {})
            if s := acc.get("species"):
                organisms.add(s)
            gene_value = acc.get("gene")
            if isinstance(gene_value, str) and (g := gene_value.strip()):
                gene_names.add(g)
            if m := xref.get("modifications"):
                modifications.extend(m)
            if b := acc.get("biotype"):
                so_terms.add(b)

        def format_unique_values(values: Set[str]) -> Optional[str]:
            if not values:
                return None
            if len(values) == 1:
                return next(iter(values))
            return ", ".join(sorted(values))

        xrefs_info = {
            "organism": format_unique_values(organisms),
            "gene_name": format_unique_values(gene_names),
            "related_genes": list(gene_names) if gene_names else None,
            "modifications": modifications or None,
            "so_term": format_unique_values(so_terms),
        }

        fallback_rules = {
            "organism": ["organism", "species"],
            "related_genes": ["related_genes", "genes"],
            "gene_name": ["gene_name", "gene"],
            "so_term": ["so_term"],
            "modifications": ["modifications"],
        }

        def resolve_field(field_name: str) -> Any:
            if (value := xrefs_info.get(field_name)) is not None:
                return value

            for key in fallback_rules[field_name]:
                if (value := rna_data.get(key)) is not None:
                    return value

            return None

        organism = resolve_field("organism")
        gene_name = resolve_field("gene_name")
        so_term = resolve_field("so_term")
        modifications = resolve_field("modifications")

        related_genes = resolve_field("related_genes")
        if not related_genes and (single_gene := rna_data.get("gene_name")):
            related_genes = [single_gene]

        sequence = rna_data.get("sequence", "")

        return {
            "molecule_type": "RNA",
            "database": "RNAcentral",
            "id": rna_id,
            "rnacentral_id": rna_data.get("rnacentral_id", rna_id),
            "sequence": sequence,
            "sequence_length": rna_data.get("length", len(sequence)),
            "rna_type": rna_data.get("rna_type", "N/A"),
            "description": rna_data.get("description", "N/A"),
            "url": f"https://rnacentral.org/rna/{rna_id}",
            "organism": organism,
            "related_genes": related_genes or None,
            "gene_name": gene_name,
            "so_term": so_term,
            "modifications": modifications,
        }

    @staticmethod
    def _calculate_md5(sequence: str) -> str:
        """
        Calculate MD5 hash for RNA sequence as per RNAcentral spec.
        - Replace U with T
        - Convert to uppercase
        - Encode as ASCII
        """
        # Normalize sequence
        normalized_seq = sequence.replace("U", "T").replace("u", "t").upper()
        if not re.fullmatch(r"[ATCGN]+", normalized_seq):
            raise ValueError(f"Invalid sequence characters after normalization: {normalized_seq[:50]}...")

        return hashlib.md5(normalized_seq.encode("ascii")).hexdigest()

    def get_by_rna_id(self, rna_id: str) -> Optional[dict]:
        """
        Get RNA information by RNAcentral ID.
        :param rna_id: RNAcentral ID (e.g., URS0000000001).
        :return: A dictionary containing RNA information or None if not found.
        """
        try:
            url = f"{self.base_url}/rna/{rna_id}"
            url += "?flat=true"

            resp = requests.get(url, headers=self.headers, timeout=self.api_timeout)
            resp.raise_for_status()

            rna_data = resp.json()
            xrefs_data = rna_data.get("xrefs", [])
            result = self._rna_data_to_dict(rna_id, rna_data, xrefs_data)
            return result
        except requests.Timeout as e:
            self.logger.warning("Timeout getting RNA ID %s (timeout=%ds): %s", rna_id, self.api_timeout, e)
            return None
        except requests.RequestException as e:
            self.logger.error("Network error getting RNA ID %s: %s", rna_id, e)
            return None
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Unexpected error getting RNA ID %s: %s", rna_id, e)
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.RequestException)),
        reraise=False,
    )
    def get_best_hit(self, keyword: str) -> Optional[dict]:
        """
        Search RNAcentral with a keyword and return the best hit.
        :param keyword: The search keyword (e.g., miRNA name, RNA name).
        :return: Dictionary with RNA information or None.
        """
        keyword = keyword.strip()
        if not keyword:
            self.logger.warning("Empty keyword provided to get_best_hit")
            return None

        try:
            url = f"{self.base_url}/rna"
            params = {"search": keyword, "format": "json"}
            resp = requests.get(url, params=params, headers=self.headers, timeout=self.api_timeout)
            resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])

            if not results:
                self.logger.info("No search results for keyword: %s", keyword)
                return None

            first_result = results[0]
            rna_id = first_result.get("rnacentral_id")

            if rna_id:
                detailed = self.get_by_rna_id(rna_id)
                if detailed:
                    return detailed
            self.logger.debug("Using search result data for %s", rna_id or "unknown")
            return self._rna_data_to_dict(rna_id or "", first_result)

        except requests.RequestException as e:
            self.logger.error("Network error searching keyword '%s': %s", keyword, e)
            return None
        except Exception as e:
            self.logger.error("Unexpected error searching keyword '%s': %s", keyword, e)
            return None

    def _local_blast(self, seq: str, threshold: float) -> Optional[str]:
        """
        Perform local BLAST search using local BLAST database.
        Optimized with multi-threading and faster output format.
        """
        try:
            # Use temporary file for query sequence
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".fa", delete=False) as tmp:
                tmp.write(f">query\n{seq}\n")
                tmp_name = tmp.name

            # Optimized BLAST command with:
            # - num_threads: Use multiple threads for faster search
            # - outfmt 6 sacc: Only return accession (minimal output)
            # - max_target_seqs 1: Only need the best hit
            # - evalue: Threshold for significance
            cmd = [
                "blastn", "-db", self.local_blast_db, "-query", tmp_name,
                "-evalue", str(threshold),
                "-max_target_seqs", "1",
                "-num_threads", str(self.blast_num_threads),
                "-outfmt", "6 sacc"  # Only accession, tab-separated
            ]
            self.logger.debug("Running local blastn for RNA (threads=%d): %s",
                        self.blast_num_threads, " ".join(cmd))

            # Run BLAST with timeout to avoid hanging
            try:
                out = subprocess.check_output(
                    cmd,
                    text=True,
                    timeout=300,  # 5 minute timeout for BLAST search
                    stderr=subprocess.DEVNULL  # Suppress BLAST warnings to reduce I/O
                ).strip()
            except subprocess.TimeoutExpired:
                self.logger.warning("BLAST search timed out after 5 minutes for sequence")
                os.remove(tmp_name)
                return None

            os.remove(tmp_name)
            return out.split("\n", maxsplit=1)[0] if out else None
        except Exception as exc:
            self.logger.error("Local blastn failed: %s", exc)
            # Clean up temp file if it still exists
            try:
                if 'tmp_name' in locals():
                    os.remove(tmp_name)
            except Exception:
                pass
            return None

    def get_by_fasta(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:  # pylint: disable=too-many-return-statements
        """
        Search RNAcentral with an RNA sequence.
        Tries local BLAST first if enabled, falls back to RNAcentral API.
        Unified approach: Find RNA ID from sequence search, then call get_by_rna_id() for complete information.
        :param sequence: RNA sequence (FASTA format or raw sequence).
        :param threshold: E-value threshold for BLAST search.
        :return: A dictionary containing complete RNA information or None if not found.
        """
        def _extract_sequence(sequence: str) -> Optional[str]:
            """Extract and normalize RNA sequence from input."""
            if sequence.startswith(">"):
                seq_lines = sequence.strip().split("\n")
                seq = "".join(seq_lines[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")
            # Accept both U (original RNA) and T
            return seq if seq and re.fullmatch(r"[AUCGTN\s]+", seq, re.I) else None

        try:
            seq = _extract_sequence(sequence)
            if not seq:
                self.logger.error("Empty or invalid RNA sequence provided.")
                return None

            # Try local BLAST first if enabled
            if self.use_local_blast:
                accession = self._local_blast(seq, threshold)
                if accession:
                    self.logger.debug("Local BLAST found accession: %s", accession)
                    detailed = self.get_by_rna_id(accession)
                    if detailed:
                        return detailed
                    self.logger.info(
                        "Local BLAST found accession %s but could not retrieve metadata from API.",
                        accession
                    )
                    return None
                self.logger.info(
                    "Local BLAST found no match for sequence. "
                    "API fallback disabled when using local database."
                )
                return None

            # Fall back to RNAcentral API only if local BLAST is not enabled
            self.logger.debug("Falling back to RNAcentral API.")

            md5_hash = self._calculate_md5(seq)
            search_url = f"{self.base_url}/rna"
            params = {"md5": md5_hash, "format": "json"}

            resp = requests.get(search_url, params=params, headers=self.headers, timeout=60)
            resp.raise_for_status()

            search_results = resp.json()
            results = search_results.get("results", [])

            if not results:
                self.logger.info("No exact match found in RNAcentral for sequence")
                return None

            rna_id = results[0].get("rnacentral_id")
            if rna_id:
                detailed = self.get_by_rna_id(rna_id)
                if detailed:
                    return detailed
                # Fallback: use search result data if get_by_rna_id returns None
                self.logger.debug("Using search result data for %s (get_by_rna_id returned None)", rna_id)
                return self._rna_data_to_dict(rna_id, results[0])

            self.logger.error("No RNAcentral ID found in search results.")
            return None
        except Exception as e:
            self.logger.error("Sequence search failed: %s", e)
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def search(self, query: str, threshold: float = 0.1, **kwargs) -> Optional[Dict]:
        """Search RNAcentral with either an RNAcentral ID, keyword, or RNA sequence."""
        if not query or not isinstance(query, str):
            self.logger.error("Empty or non-string input.")
            return None

        query = query.strip()
        self.logger.debug("RNAcentral search query: %s", query)

        loop = asyncio.get_running_loop()

        # check if RNA sequence (AUCG or ATCG characters, contains U or T)
        # Note: Sequences with T are also RNA sequences
        is_rna_sequence = query.startswith(">") or (
            re.fullmatch(r"[AUCGTN\s]+", query, re.I) and
            ("U" in query.upper() or "T" in query.upper())
        )
        if is_rna_sequence:
            result = await loop.run_in_executor(_get_pool(), self.get_by_fasta, query, threshold)
        # check if RNAcentral ID (typically starts with URS)
        elif re.fullmatch(r"URS\d+", query, re.I):
            result = await loop.run_in_executor(_get_pool(), self.get_by_rna_id, query)
        else:
            # otherwise treat as keyword
            result = await loop.run_in_executor(_get_pool(), self.get_best_hit, query)

        if result:
            result["_search_query"] = query
        return result
