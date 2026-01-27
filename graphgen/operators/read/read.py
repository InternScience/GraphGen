from pathlib import Path
from typing import Any, List, Optional, Union

import ray

from graphgen.common import init_storage
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
    working_dir: Optional[str] = "cache",
    kv_backend: str = "rocksdb",
    parallelism: int = 4,
    recursive: bool = True,
    read_nums: Optional[int] = None,
    **reader_kwargs: Any,
) -> ray.data.Dataset:
    """
    Unified entry point to read files of multiple types using Ray Data.

    :param input_path: File or directory path(s) to read from
    :param allowed_suffix: List of allowed file suffixes (e.g., ['pdf', 'txt'])
    :param working_dir: Directory to cache intermediate files (PDF processing)
    :param kv_backend: Backend for key-value storage
    :param parallelism: Number of parallel workers
    :param recursive: Whether to scan directories recursively
    :param read_nums: Limit the number of documents to read
    :param reader_kwargs: Additional kwargs passed to readers
    :return: Ray Dataset containing all documents
    """

    read_cache = init_storage(
        backend=kv_backend, working_dir=working_dir, namespace="read"
    )
    try:
        # 1. Scan all paths to discover files
        logger.info("[READ] Scanning paths: %s", input_path)
        with ParallelFileScanner(
            read_cache=read_cache,
            allowed_suffix=allowed_suffix,
            rescan=False,
            max_workers=parallelism if parallelism > 0 else 1,
        ) as scanner:
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
                reader = _build_reader(suffix, working_dir, **reader_kwargs)
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
                    "_trace_id": compute_mm_hash(record, prefix="doc-"),
                }
            )

            if read_nums is not None:
                combined_ds = combined_ds.limit(read_nums)

            # sample record
            for i, item in enumerate(combined_ds.take(1)):
                logger.debug("[READ] Sample record %d: %s", i, item)

            logger.info("[READ] Successfully read files from %s", input_path)
            return combined_ds

    except Exception as e:
        logger.error("[READ] Failed to read files from %s: %s", input_path, e)
        raise
