import re
from typing import List, Optional

from graphgen.bases.base_splitter import BaseSplitter
from graphgen.utils.log import logger


class SequenceSplitter(BaseSplitter):
    """
    Splitter for biological sequences (DNA, RNA, protein).
    Supports chunking by fixed length with overlap.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        length_function=None,
        **kwargs,
    ):
        """
        Initialize sequence splitter.
        
        :param chunk_size: Maximum length of each chunk (in sequence characters)
        :param chunk_overlap: Number of characters to overlap between chunks
        :param length_function: Function to calculate length (default: len)
        """
        if length_function is None:
            length_function = len
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            **kwargs,
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split a sequence into chunks of fixed size with overlap.
        
        :param text: The sequence to split (may include FASTA header)
        :return: List of sequence chunks
        """
        # Remove FASTA header if present
        sequence = self._extract_sequence(text)
        
        if not sequence:
            logger.warning("Empty sequence provided to SequenceSplitter")
            return []
        
        # If sequence is shorter than chunk_size, return as single chunk
        if len(sequence) <= self.chunk_size:
            return [sequence]
        
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        
        while start < len(sequence):
            end = min(start + self.chunk_size, len(sequence))
            chunk = sequence[start:end]
            chunks.append(chunk)
            start += step
            
            # Avoid infinite loop if step is 0 or negative
            if step <= 0:
                break
        
        return chunks

    @staticmethod
    def _extract_sequence(text: str) -> str:
        """
        Extract sequence from text, removing FASTA headers and whitespace.
        
        :param text: Input text (may contain FASTA header)
        :return: Clean sequence string
        """
        # Remove FASTA header lines (lines starting with >)
        lines = text.split("\n")
        sequence_lines = [line for line in lines if not line.strip().startswith(">")]
        
        # Join and remove whitespace
        sequence = "".join(sequence_lines)
        sequence = re.sub(r"\s+", "", sequence)
        
        return sequence.upper()  # Normalize to uppercase

    @staticmethod
    def detect_sequence_type(sequence: str) -> Optional[str]:
        """
        Detect the type of sequence (DNA, RNA, or protein).
        
        :param sequence: The sequence string
        :return: "dna", "rna", "protein", or None if cannot determine
        """
        # Remove FASTA header and whitespace
        clean_seq = SequenceSplitter._extract_sequence(sequence)
        
        if not clean_seq:
            return None
        
        # Check for protein-specific amino acids
        protein_chars = set("EFILPQXZ")  # Amino acids not in DNA/RNA
        if any(char in clean_seq for char in protein_chars):
            return "protein"
        
        # Check for RNA-specific character (U)
        if "U" in clean_seq.upper():
            return "rna"
        
        # Check if contains only DNA/RNA characters (A, T, G, C, N)
        dna_rna_chars = set("ATGCUN")
        if all(char.upper() in dna_rna_chars for char in clean_seq):
            # Default to DNA if ambiguous (could be DNA or RNA without U)
            return "dna"
        
        # If contains other characters, might be protein
        return "protein"
