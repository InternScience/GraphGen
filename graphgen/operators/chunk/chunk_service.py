import os
from functools import lru_cache
from typing import Union, Tuple

from graphgen.bases import BaseOperator
from graphgen.models import (
    ChineseRecursiveTextSplitter,
    RecursiveCharacterSplitter,
    Tokenizer,
)
from graphgen.utils import detect_main_language

_MAPPING = {
    "en": RecursiveCharacterSplitter,
    "zh": ChineseRecursiveTextSplitter,
}

SplitterT = Union[RecursiveCharacterSplitter, ChineseRecursiveTextSplitter]


@lru_cache(maxsize=None)
def _get_splitter(language: str, frozen_kwargs: frozenset) -> SplitterT:
    cls = _MAPPING[language]
    kwargs = dict(frozen_kwargs)
    return cls(**kwargs)


def split_chunks(text: str, language: str = "en", **kwargs) -> list:
    if language not in _MAPPING:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages are: {list(_MAPPING.keys())}"
        )
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
        tokenizer_model = os.getenv("TOKENIZER_MODEL", "cl100k_base")
        self.tokenizer_instance: Tokenizer = Tokenizer(model_name=tokenizer_model)
        self.chunk_kwargs = chunk_kwargs

    def process(self, batch: list) -> Tuple[list, dict]:
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
                        "length": len(self.tokenizer_instance.encode(text_chunk))
                        if self.tokenizer_instance
                        else len(text_chunk),
                        "language": doc_language,
                    }
                    chunk["_trace_id"] = self.get_trace_id(chunk)
                    results.append(chunk)
                    meta_updates.setdefault(doc["_trace_id"], []).append(
                        chunk["_trace_id"]
                    )
            else:
                # other types of documents(images, sequences) are not chunked
                data = doc.copy()
                data["_trace_id"] = self.get_trace_id(data)
                results.append(data)
                meta_updates.setdefault(doc["_trace_id"], []).append(data["_trace_id"])
        return results, meta_updates
