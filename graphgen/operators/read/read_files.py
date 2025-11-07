from pathlib import Path
from typing import Any, Dict, List

from graphgen.models import (
    CSVReader,
    JSONLReader,
    JSONReader,
    ParquetReader,
    PDFReader,
    PickleReader,
    RDFReader,
    TXTReader,
)
from graphgen.utils import logger

_MAPPING = {
    "jsonl": JSONLReader,
    "json": JSONReader,
    "txt": TXTReader,
    "csv": CSVReader,
    "pdf": PDFReader,
    "parquet": ParquetReader,
    "pickle": PickleReader,
    "rdf": RDFReader,
    "owl": RDFReader,
    "ttl": RDFReader,
}


def _build_reader(suffix: str, cache_dir: str | None):
    suffix = suffix.lower()
    if suffix == "pdf" and cache_dir is not None:
        return _MAPPING[suffix](output_dir=cache_dir)
    return _MAPPING[suffix]()


def read_files(file_path: str, cache_dir: str | None = None) -> list[dict]:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input_path not found: {file_path}")

    if path.is_file():
        suffix = path.suffix.lstrip(".")
        reader = _build_reader(suffix, cache_dir)
        return reader.read(str(path))

    support_suffix = set(_MAPPING.keys())
    files_to_read = [
        p for p in path.rglob("*") if p.suffix.lstrip(".").lower() in support_suffix
    ]
    logger.info("Found %d file(s) under folder %s", len(files_to_read), file_path)

    all_docs: List[Dict[str, Any]] = []
    for p in files_to_read:
        try:
            suffix = p.suffix.lstrip(".")
            reader = _build_reader(suffix, cache_dir)
            all_docs.extend(reader.read(str(p)))
        except Exception as e:  # pylint: disable=broad-except
            logger.exception("Error reading %s: %s", p, e)

    return all_docs
