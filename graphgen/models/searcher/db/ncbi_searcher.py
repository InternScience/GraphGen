import asyncio
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.client import IncompleteRead
from typing import Dict, Optional

from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from requests.exceptions import RequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher
>>>>>>> feature/multi-omics-qa


@lru_cache(maxsize=None)
def _get_pool():

        if not representative_accession:
            representative_accession = next(
                (
                    product.get("Gene-commentary_accession")
                    for product in locus.get("Gene-commentary_products", [])
                    if product.get("Gene-commentary_accession")
                ),
                None,
            )

        # Extract function
        function = data.get("Entrezgene_summary") or next(
            (
                comment.get("Gene-commentary_comment")
                for comment in data.get("Entrezgene_comments", [])
                if isinstance(comment, dict)
                and "function" in str(comment.get("Gene-commentary_heading", "")).lower()
            ),
            None,
        )

        return {
            "molecule_type": "DNA",
            "database": "NCBI",
            "id": gene_id,
            "gene_name": gene_ref.get("Gene-ref_locus", "N/A"),
            "gene_description": gene_ref.get("Gene-ref_desc", "N/A"),
            "organism": self._nested_get(
                biosource, "BioSource_org", "Org-ref", "Org-ref_taxname", default="N/A"
            ),
            "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
            "gene_synonyms": gene_synonyms or None,
            "gene_type": {
                "1": "protein-coding",
                "2": "pseudo",
                "3": "rRNA",
                "4": "tRNA",
                "5": "snRNA",
                "6": "ncRNA",
                "7": "other",
            }.get(str(data.get("Entrezgene_type")), f"type_{data.get('Entrezgene_type')}"),
            "chromosome": chromosome_match.group(1) if chromosome_match else None,
            "genomic_location": genomic_location,
            "function": function,
            # Fields from accession-based queries
            "title": None,
            "sequence": None,
            "sequence_length": None,
            "gene_id": gene_id,
            "molecule_type_detail": self._infer_molecule_type_detail(
                representative_accession, data.get("Entrezgene_type")
            ),
            "_representative_accession": representative_accession,
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
    )
    def get_by_gene_id(self, gene_id: str, preferred_accession: Optional[str] = None) -> Optional[dict]:
        """Get gene information by Gene ID."""
        def _extract_metadata_from_genbank(result: dict, accession: str):
            """Extract metadata from GenBank format (title, features, organism, etc.)."""
            with Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text") as handle:
                record = SeqIO.read(handle, "genbank")

                result["title"] = record.description
                result["molecule_type_detail"] = self._infer_molecule_type_detail(accession) or "N/A"

                for feature in record.features:
                    if feature.type == "source":
                        if 'chromosome' in feature.qualifiers:
                            result["chromosome"] = feature.qualifiers['chromosome'][0]

                        if feature.location:
                            start = int(feature.location.start) + 1
                            end = int(feature.location.end)
                            result["genomic_location"] = f"{start}-{end}"

                        break

                if not result.get("organism") and 'organism' in record.annotations:
                    result["organism"] = record.annotations['organism']

            return result

        def _extract_sequence_from_fasta(result: dict, accession: str):
            """Extract sequence from FASTA format (more reliable than GenBank for CON-type records)."""
            try:
                with Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text") as fasta_handle:
                    fasta_record = SeqIO.read(fasta_handle, "fasta")
                    result["sequence"] = str(fasta_record.seq)
                    result["sequence_length"] = len(fasta_record.seq)
            except Exception as fasta_exc:
                self.logger.warning(
                    "Failed to extract sequence from accession %s using FASTA format: %s",
                    accession, fasta_exc
                )
                result["sequence"] = None
                result["sequence_length"] = None
            return result

        def _extract_sequence(result: dict, accession: str):
            """
            Extract sequence using the appropriate method based on configuration.
            If use_local_blast=True, use local database. Otherwise, use NCBI API.
            Always fetches sequence (no option to skip).
            """
            # If using local BLAST, use local database
            if self.use_local_blast:
                sequence = self._extract_sequence_from_local_db(accession)

                if sequence:
                    result["sequence"] = sequence
                    result["sequence_length"] = len(sequence)
                else:
                    # Failed to extract from local DB, set to None (no fallback to API)
                    result["sequence"] = None
                    result["sequence_length"] = None
                    self.logger.warning(
                        "Failed to extract sequence from local DB for accession %s. "
                        "Not falling back to NCBI API as use_local_blast=True.",
                        accession
                    )
            else:
                # Use NCBI API to fetch sequence
                result = _extract_sequence_from_fasta(result, accession)

            return result

        try:
            with Entrez.efetch(db="gene", id=gene_id, retmode="xml") as handle:
                gene_record = Entrez.read(handle)

            if not gene_record:
                return None

            result = self._gene_record_to_dict(gene_record, gene_id)

            if accession := (preferred_accession or result.get("_representative_accession")):
                result = _extract_metadata_from_genbank(result, accession)
                # Extract sequence using appropriate method
                result = _extract_sequence(result, accession)

            result.pop("_representative_accession", None)
            return result
        except (RequestException, IncompleteRead):
            raise
        except Exception as exc:
            self.logger.error("Gene ID %s not found: %s", gene_id, exc)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
    )
    def get_by_accession(self, accession: str) -> Optional[dict]:
        """Get sequence information by accession number."""
        def _extract_gene_id(link_handle):
            """Extract GeneID from elink results."""
            links = Entrez.read(link_handle)
            if not links or "LinkSetDb" not in links[0]:
                return None

            for link_set in links[0]["LinkSetDb"]:
                if link_set.get("DbTo") != "gene":
                    continue

                link = (link_set.get("Link") or link_set.get("IdList", [{}]))[0]
                return str(link.get("Id") if isinstance(link, dict) else link)

        try:
            # TODO: support accession number with version number (e.g., NM_000546.3)
            with Entrez.elink(dbfrom="nuccore", db="gene", id=accession) as link_handle:
                gene_id = _extract_gene_id(link_handle)

            if not gene_id:
                self.logger.warning("Accession %s has no associated GeneID", accession)
                return None

            result = self.get_by_gene_id(gene_id, preferred_accession=accession)

            if result:
                result["id"] = accession
                result["url"] = f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}"

            return result
        except (RequestException, IncompleteRead):
            raise
        except Exception as exc:
            self.logger.error("Accession %s not found: %s", accession, exc)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
    )
    def get_best_hit(self, keyword: str) -> Optional[dict]:
        """Search NCBI Gene database with a keyword and return the best hit."""
        if not keyword.strip():
            return None

        try:
            for search_term in [f"{keyword}[Gene] OR {keyword}[All Fields]", keyword]:
                with Entrez.esearch(db="gene", term=search_term, retmax=1, sort="relevance") as search_handle:
                    search_results = Entrez.read(search_handle)

                if len(gene_id := search_results.get("IdList", [])) > 0:
                    result = self.get_by_gene_id(gene_id[0])
                    return result
        except (RequestException, IncompleteRead):
            raise
        except Exception as e:
            self.logger.error("Keyword %s not found: %s", keyword, e)
        return None

    def _extract_sequence_from_local_db(self, accession: str) -> Optional[str]:
        """Extract sequence from local BLAST database using blastdbcmd."""
        try:
            cmd = [
                "blastdbcmd",
                "-db", self.local_blast_db,
                "-entry", accession,
                "-outfmt", "%s"  # Only sequence, no header
            ]
            sequence = subprocess.check_output(
                cmd,
                text=True,
                timeout=10,  # 10 second timeout for local extraction
                stderr=subprocess.DEVNULL
            ).strip()
            return sequence if sequence else None
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout extracting sequence from local DB for accession %s", accession)
            return None
        except Exception as exc:
            self.logger.warning("Failed to extract sequence from local DB for accession %s: %s", accession, exc)
            return None

    def _local_blast(self, seq: str, threshold: float) -> Optional[str]:
        """
        Perform local BLAST search using local BLAST database.
        Optimized with multi-threading and faster output format.
        """
        try:
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
            self.logger.debug("Running local blastn (threads=%d): %s",
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

    def get_by_fasta(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:
        """Search NCBI with a DNA sequence using BLAST."""

        def _extract_and_normalize_sequence(sequence: str) -> Optional[str]:
            """Extract and normalize DNA sequence from input."""
            if sequence.startswith(">"):
                seq = "".join(sequence.strip().split("\n")[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")
            return seq if re.fullmatch(r"[ATCGN]+", seq, re.I) else None


        def _process_network_blast_result(blast_record, seq: str, threshold: float) -> Optional[dict]:
            """Process network BLAST result and return dictionary or None."""
            if not blast_record.alignments:
                self.logger.info("No BLAST hits found for the given sequence.")
                return None

            best_alignment = blast_record.alignments[0]
            best_hsp = best_alignment.hsps[0]
            if best_hsp.expect > threshold:
                self.logger.info("No BLAST hits below the threshold E-value.")
                return None

            hit_id = best_alignment.hit_id
            if accession_match := re.search(r"ref\|([^|]+)", hit_id):
                return self.get_by_accession(accession_match.group(1).split(".")[0])

            # If unable to extract accession, return basic information
            return {
                "molecule_type": "DNA",
                "database": "NCBI",
                "id": hit_id,
                "title": best_alignment.title,
                "sequence_length": len(seq),
                "e_value": best_hsp.expect,
                "identity": best_hsp.identities / best_hsp.align_length if best_hsp.align_length > 0 else 0,
                "url": f"https://www.ncbi.nlm.nih.gov/nuccore/{hit_id}",
            }

        try:
            if not (seq := _extract_and_normalize_sequence(sequence)):
                self.logger.error("Empty or invalid DNA sequence provided.")
                return None

            # Try local BLAST first if enabled
            if self.use_local_blast:
                accession = self._local_blast(seq, threshold)

                if accession:
                    self.logger.debug("Local BLAST found accession: %s", accession)
                    # When using local BLAST, skip sequence fetching by default (faster, fewer API calls)
                    # Sequence is already known from the query, so we only need metadata
                    result = self.get_by_accession(accession)
                    return result

                self.logger.info(
                    "Local BLAST found no match for sequence. "
                    "API fallback disabled when using local database."
                )
                return None

            # Fall back to network BLAST only if local BLAST is not enabled
            self.logger.debug("Falling back to NCBIWWW.qblast")
            with NCBIWWW.qblast("blastn", "nr", seq, hitlist_size=1, expect=threshold) as result_handle:
                result = _process_network_blast_result(NCBIXML.read(result_handle), seq, threshold)
            return result
        except (RequestException, IncompleteRead):
            raise
        except Exception as e:
            self.logger.error("BLAST search failed: %s", e)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
    )
    async def search(self, query: str, threshold: float = 0.01, **kwargs) -> Optional[Dict]:
        """Search NCBI with either a gene ID, accession number, keyword, or DNA sequence."""
        if not query or not isinstance(query, str):
            self.logger.error("Empty or non-string input.")
            return None

        query = query.strip()
        self.logger.debug("NCBI search query: %s", query)

        loop = asyncio.get_running_loop()

        # Auto-detect query type and execute in thread pool
        # All methods need lock because they all call NCBI API (rate limit: max 3 requests per second)
        # Even if get_by_fasta uses local BLAST, it still calls get_by_accession which needs API
        async def _execute_with_lock(func, *args):
            """Execute function with lock for NCBI API calls."""
            async with _blast_lock:
                return await loop.run_in_executor(_get_pool(), func, *args)

        if query.startswith(">") or re.fullmatch(r"[ATCGN\s]+", query, re.I):
            # FASTA sequence: always use lock (even with local BLAST, get_by_accession needs API)
            result = await _execute_with_lock(self.get_by_fasta, query, threshold)
        elif re.fullmatch(r"^\d+$", query):
            # Gene ID: always use lock (network API call)
            result = await _execute_with_lock(self.get_by_gene_id, query)
        elif re.fullmatch(r"[A-Z]{2}_\d+\.?\d*", query, re.I):
            # Accession: always use lock (network API call)
            result = await _execute_with_lock(self.get_by_accession, query)
        else:
            # Keyword: always use lock (network API call)
            result = await _execute_with_lock(self.get_best_hit, query)

        if result:
            result["_search_query"] = query
        return result
