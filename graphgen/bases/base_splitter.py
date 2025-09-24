import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from graphgen.bases.datatypes import Chunk


@dataclass
class BaseSplitter(ABC):
    """
    Abstract base class for splitting text into smaller chunks.
    """

    chunk_size: int = 1024
    chunk_overlap_size: int = 100
    length_function: Callable[[str], int] = len
    keep_separator: bool = False
    add_start_index: bool = False

    @abstractmethod
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split the input text into smaller chunks.

        :param text: The input text to be split.
        :return: A list of dictionaries, each containing a chunk of text and optionally its start index.
        """

    def create_chunks(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Chunk]:
        """
        Turn a list of texts into a list of Chunks, with optional metadata.
        :param texts:
        :param metadatas:
        :return:
        """
        _metadatas = metadatas or [{}] * len(texts)
        chunks = []
        for i, text in enumerate(texts):
            chunks.append(Chunk(content=text, metadata=copy.deepcopy(_metadatas[i])))
        return chunks

    def split(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        texts = self.split_text(text)
        return self.create_chunks(texts, [metadata] * len(texts) if metadata else None)
