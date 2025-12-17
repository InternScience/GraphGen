from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import ray

from graphgen.models import (
    CSVReader,
    JSONReader,
    ParquetReader,
    PDFReader,
    PickleReader,
    RDFReader,
    TXTReader,
)
from graphgen.utils import compute_mm_hash, logger

from .parallel_file_scanner import ParallelFileScanner

_MAPPING = {
    "jsonl": JSONReader,
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


def _build_reader(suffix: str, cache_dir: str | None, **reader_kwargs):
    """Factory function to build appropriate reader instance"""
    suffix = suffix.lower()
    reader_cls = _MAPPING.get(suffix)
    if not reader_cls:
        raise ValueError(f"Unsupported file suffix: {suffix}")

    # Special handling for PDFReader which needs output_dir
    if suffix == "pdf":
        if cache_dir is None:
            raise ValueError("cache_dir must be provided for PDFReader")
        return reader_cls(output_dir=cache_dir, **reader_kwargs)

    return reader_cls(**reader_kwargs)


def read(
    input_path: Union[str, List[str]],
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = "cache",
    parallelism: int = 4,
    recursive: bool = True,
    **reader_kwargs: Any,
) -> ray.data.Dataset:
    """
    Unified entry point to read files of multiple types using Ray Data.

    :param input_path: File or directory path(s) to read from
    :param allowed_suffix: List of allowed file suffixes (e.g., ['pdf', 'txt'])
    :param cache_dir: Directory to cache intermediate files (PDF processing)
    :param parallelism: Number of parallel workers
    :param recursive: Whether to scan directories recursively
    :param reader_kwargs: Additional kwargs passed to readers
    :return: Ray Dataset containing all documents
    """
    try:
        # 1. Scan all paths to discover files
        logger.info("[READ] Scanning paths: %s", input_path)
        scanner = ParallelFileScanner(
            cache_dir=cache_dir,
            allowed_suffix=allowed_suffix,
            rescan=False,
            max_workers=parallelism if parallelism > 0 else 1,
        )

        all_files = []
        scan_results = scanner.scan(input_path, recursive=recursive)

        for result in scan_results.values():
            all_files.extend(result.get("files", []))

        logger.info("[READ] Found %d files to process", len(all_files))

        if not all_files:
            raise ValueError("No files found to read.")

        # 2. Group files by suffix to use appropriate reader
        files_by_suffix = {}
        for file_info in all_files:
            suffix = Path(file_info["path"]).suffix.lower().lstrip(".")
            if allowed_suffix and suffix not in [
                s.lower().lstrip(".") for s in allowed_suffix
            ]:
                continue
            files_by_suffix.setdefault(suffix, []).append(file_info["path"])

        # 3. Create read tasks
        read_tasks = []
        for suffix, file_paths in files_by_suffix.items():
            reader = _build_reader(suffix, cache_dir, **reader_kwargs)
            ds = reader.read(file_paths)
            read_tasks.append(ds)

        # 4. Combine all datasets
        if not read_tasks:
            raise ValueError("No datasets created from the provided files.")

        if len(read_tasks) == 1:
            combined_ds = read_tasks[0]
        else:
            combined_ds = read_tasks[0].union(*read_tasks[1:])

        combined_ds = combined_ds.map(
            lambda record: {
                **record,
                "_doc_id": compute_mm_hash(record, prefix="doc-"),
            }
        )

        logger.info("[READ] Successfully read files from %s", input_path)
        return combined_ds

    except Exception as e:
        logger.error("[READ] Failed to read files from %s: %s", input_path, e)
        raise


def read_files(
    input_file: str,
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    max_workers: int = 4,
    rescan: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Read files from a path using parallel scanning and appropriate readers.
    Returns an iterator for streaming (backward compatibility with graphgen.py).

    Args:
        input_file: Path to a file or directory
        allowed_suffix: List of file suffixes to read. If None, uses all supported types
        cache_dir: Directory for caching PDF extraction and scan results
        max_workers: Number of workers for parallel scanning
        rescan: Whether to force rescan even if cached results exist

    Returns:
        Iterator of dictionaries containing the data (for streaming)
    """
    path = Path(input_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input_path not found: {input_file}")

    if allowed_suffix is None:
        support_suffix = set(_MAPPING.keys())
    else:
        support_suffix = {s.lower().lstrip(".") for s in allowed_suffix}

    with ParallelFileScanner(
        cache_dir=cache_dir or "cache",
        allowed_suffix=support_suffix,
        rescan=rescan,
        max_workers=max_workers,
    ) as scanner:
        scan_results = scanner.scan(str(path), recursive=True)

    # Extract files from scan results
    files_to_read = []
    for path_result in scan_results.values():
        if "error" in path_result:
            logger.warning("Error scanning %s: %s", path_result.path, path_result.error)
            continue
        files_to_read.extend(path_result.get("files", []))

    logger.info(
        "Found %d eligible file(s) under folder %s (allowed_suffix=%s)",
        len(files_to_read),
        input_file,
        support_suffix,
    )

    for file_info in files_to_read:
        try:
            file_path = file_info["path"]
            suffix = Path(file_path).suffix.lstrip(".").lower()
            reader = _build_reader(suffix, cache_dir)

            # Prefer stream reading if available (for memory efficiency)
            if hasattr(reader, "read_stream"):
                yield from reader.read_stream(file_path)
            else:
                # Fallback to regular read() method - convert Ray Dataset to iterator
                ds = reader.read([file_path])
                yield from ds.iter_rows()

        except Exception as e:  # pylint: disable=broad-except
            logger.exception("Error reading %s: %s", file_info.get("path"), e)
