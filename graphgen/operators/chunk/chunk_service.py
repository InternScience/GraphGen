import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from graphgen.bases import BaseOperator
from graphgen.utils import detect_main_language

if TYPE_CHECKING:
    from graphgen.models import (
        ChineseRecursiveTextSplitter,
        RecursiveCharacterSplitter,
        Tokenizer,
    )

if TYPE_CHECKING:
    SplitterT = Union["RecursiveCharacterSplitter", "ChineseRecursiveTextSplitter"]
else:
    SplitterT = Any


@lru_cache(maxsize=None)
def _get_splitter(language: str, frozen_kwargs: frozenset) -> SplitterT:
    kwargs = dict(frozen_kwargs)
    if language == "en":
        from graphgen.models import RecursiveCharacterSplitter

        return RecursiveCharacterSplitter(**kwargs)
    if language == "zh":
        from graphgen.models import ChineseRecursiveTextSplitter

        return ChineseRecursiveTextSplitter(**kwargs)
    raise ValueError(
        f"Unsupported language: {language}. Supported languages are: en, zh"
    )


def split_chunks(text: str, language: str = "en", **kwargs) -> list:
    frozen_kwargs = frozenset(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()
    )
    splitter = _get_splitter(language, frozen_kwargs)
    return splitter.split_text(text)


class ChunkService(BaseOperator):
    def __init__(
        self, working_dir: str = "cache", kv_backend: str = "rocksdb", **chunk_kwargs
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="chunk"
        )
        self.tokenizer_model = os.getenv("TOKENIZER_MODEL", "cl100k_base")
        self._tokenizer_instance: Optional["Tokenizer"] = None
        self.chunk_kwargs = chunk_kwargs

    @property
    def tokenizer_instance(self) -> "Tokenizer":
        if self._tokenizer_instance is None:
            from graphgen.models import Tokenizer

            self._tokenizer_instance = Tokenizer(model_name=self.tokenizer_model)
        return self._tokenizer_instance

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Chunk the documents in the batch.
        :return: A tuple of (results, meta_updates)
            results: A list of chunked documents. Each chunked document is a dict with the structure:
                {"_trace_id": str, "content": str, "type": str,  "metadata": {"length": int, "language": str, ...}
            meta_updates: A dict mapping source document IDs to lists of trace IDs for the chunked documents.
        """
        results = []
        meta_updates = {}
        for doc in batch:
            doc_type = doc.get("type")
            if doc_type == "text":
                doc_language = detect_main_language(doc["content"])
                text_chunks = split_chunks(
                    doc["content"],
                    language=doc_language,
                    **self.chunk_kwargs,
                )
                for text_chunk in text_chunks:
                    chunk = {
                        "content": text_chunk,
                        "type": "text",
                        "metadata": {
                            "length": len(self.tokenizer_instance.encode(text_chunk))
                            if self.tokenizer_instance
                            else len(text_chunk),
                            "language": doc_language,
                        },
                    }
                    chunk["_trace_id"] = self.get_trace_id(chunk)
                    results.append(chunk)
                    meta_updates.setdefault(doc["_trace_id"], []).append(
                        chunk["_trace_id"]
                    )
            else:
                # other types of documents(images, sequences) are not chunked
                data = doc.copy()
                input_trace_id = data.pop("_trace_id")
                content = data.pop("content") if "content" in data else ""
                doc_type = data.pop("type")
                chunk = {"content": content, "type": doc_type, "metadata": data}
                chunk["_trace_id"] = self.get_trace_id(chunk)
                results.append(chunk)
                meta_updates.setdefault(input_trace_id, []).append(chunk["_trace_id"])
        return results, meta_updates
