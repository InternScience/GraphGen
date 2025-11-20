from pathlib import Path
from typing import Iterator, List, Optional

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
    "md": TXTReader,
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


def read_files(
    input_file: str,
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> Iterator[list[dict]]:
    path = Path(input_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"[Reader] input_path not found: {input_file}")

    if allowed_suffix is None:
        support_suffix = set(_MAPPING.keys())
    else:
        support_suffix = {s.lower().lstrip(".") for s in allowed_suffix}

    # single file
    if path.is_file():
        suffix = path.suffix.lstrip(".").lower()
        if suffix not in support_suffix:
            logger.warning(
                "[Reader] Skip file %s (suffix '%s' not in allowed_suffix %s)",
                path,
                suffix,
                support_suffix,
            )
            return
        reader = _build_reader(suffix, cache_dir)
        logger.info("[Reader] Reading file %s", path)
        yield reader.read(str(path))
        return

    # folder
    logger.info("[Reader] Streaming directory %s", path)
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lstrip(".").lower() in support_suffix:
            try:
                suffix = p.suffix.lstrip(".").lower()
                reader = _build_reader(suffix, cache_dir)
                logger.info("[Reader] Reading file %s", p)
                docs = reader.read(str(p))
                if docs:
                    yield docs
            except Exception:  # pylint: disable=broad-except
                logger.exception("[Reader] Error reading %s", p)
