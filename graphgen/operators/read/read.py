from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from graphgen.common.init_storage import init_storage
from graphgen.models import (
    CSVReader,
    HuggingFaceReader,
    JSONReader,
    ParquetReader,
    PDFReader,
    PickleReader,
    RDFReader,
    TXTReader,
)
from graphgen.utils import compute_dict_hash, logger

from .parallel_file_scanner import ParallelFileScanner

if TYPE_CHECKING:
    import ray
    import ray.data


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


def _process_huggingface_datasets(hf_uris: List[str], reader_kwargs: dict) -> list:
    """Process HuggingFace datasets and return list of Ray datasets."""
    logger.info("[READ] Processing HuggingFace datasets: %s", hf_uris)
    hf_reader = HuggingFaceReader(**reader_kwargs)
    read_tasks = []
    for hf_uri in hf_uris:
        # Parse URI format: "huggingface://dataset_name:subset:split"
        uri_part = hf_uri.replace("huggingface://", "")
        ds = hf_reader.read(uri_part)
        read_tasks.append(ds)
    logger.info("[READ] Successfully loaded %d HuggingFace dataset(s)", len(hf_uris))
    return read_tasks


def _process_local_files(
    local_paths: List[str],
    allowed_suffix: Optional[List[str]],
    kv_backend: str,
    working_dir: str,
    parallelism: int,
    recursive: bool,
    reader_kwargs: dict,
) -> list:
    """Process local files and return list of Ray datasets."""
    logger.info("[READ] Scanning local paths: %s", local_paths)
    read_tasks = []
    input_path_cache = init_storage(
        backend=kv_backend, working_dir=working_dir, namespace="input_path"
    )
    with ParallelFileScanner(
        input_path_cache=input_path_cache,
        allowed_suffix=allowed_suffix,
        rescan=False,
        max_workers=parallelism if parallelism > 0 else 1,
    ) as scanner:
        all_files = []
        scan_results = scanner.scan(local_paths, recursive=recursive)

        for result in scan_results.values():
            all_files.extend(result.get("files", []))

        logger.info("[READ] Found %d files to process", len(all_files))

        if all_files:
            # Group files by suffix to use appropriate reader
            files_by_suffix = {}
            for file_info in all_files:
                suffix = Path(file_info["path"]).suffix.lower().lstrip(".")
                if allowed_suffix and suffix not in [
                    s.lower().lstrip(".") for s in allowed_suffix
                ]:
                    continue
                files_by_suffix.setdefault(suffix, []).append(file_info["path"])

            # Create read tasks for files
            for suffix, file_paths in files_by_suffix.items():
                reader = _build_reader(suffix, working_dir, **reader_kwargs)
                ds = reader.read(file_paths)
                read_tasks.append(ds)

    return read_tasks


def _combine_datasets(
    read_tasks: list,
    read_nums: Optional[int],
    read_storage,
    input_path: Union[str, List[str]],
) -> "ray.data.Dataset":
    """Combine datasets and apply post-processing."""
    combined_ds = (
        read_tasks[0] if len(read_tasks) == 1 else read_tasks[0].union(*read_tasks[1:])
    )

    if read_nums is not None:
        combined_ds = combined_ds.limit(read_nums)

    def add_trace_id(batch):
        batch["_trace_id"] = batch.apply(
            lambda row: compute_dict_hash(row, prefix="read-"), axis=1
        )
        records = batch.to_dict(orient="records")
        data_to_upsert = {record["_trace_id"]: record for record in records}
        read_storage.upsert(data_to_upsert)
        read_storage.index_done_callback()
        return batch

    combined_ds = combined_ds.map_batches(add_trace_id, batch_format="pandas")

    # sample record
    for i, item in enumerate(combined_ds.take(1)):
        logger.debug("[READ] Sample record %d: %s", i, item)

    logger.info("[READ] Successfully read data from %s", input_path)
    return combined_ds


def read(
    input_path: Union[str, List[str]],
    allowed_suffix: Optional[List[str]] = None,
    working_dir: Optional[str] = "cache",
    kv_backend: str = "rocksdb",
    parallelism: int = 4,
    recursive: bool = True,
    read_nums: Optional[int] = None,
    **reader_kwargs: Any,
) -> "ray.data.Dataset":
    """
    Unified entry point to read files of multiple types using Ray Data.
    Supports both local files and Hugging Face datasets.

    :param input_path: File or directory path(s) to read from, or HuggingFace dataset URIs
                      Format for HuggingFace: "huggingface://dataset_name:subset:split"
                      Example: "huggingface://wikitext:wikitext-103-v1:train"
    :param allowed_suffix: List of allowed file suffixes (e.g., ['pdf', 'txt'])
    :param working_dir: Directory to cache intermediate files (PDF processing)
    :param kv_backend: Backend for key-value storage
    :param parallelism: Number of parallel workers
    :param recursive: Whether to scan directories recursively
    :param read_nums: Limit the number of documents to read
    :param reader_kwargs: Additional kwargs passed to readers
    :return: Ray Dataset containing all documents
    """
    import ray

    # Convert single input_path to list for uniform processing
    if isinstance(input_path, str):
        input_paths = [input_path]
    else:
        input_paths = input_path

    # Separate HuggingFace URIs from local file paths
    hf_uris = []
    local_paths = []
    for path in input_paths:
        if isinstance(path, str) and path.startswith("huggingface://"):
            hf_uris.append(path)
        else:
            local_paths.append(path)

    read_storage = init_storage(
        backend=kv_backend, working_dir=working_dir, namespace="read"
    )

    try:
        read_tasks = []

        # 1. Process HuggingFace datasets if any
        if hf_uris:
            read_tasks.extend(_process_huggingface_datasets(hf_uris, reader_kwargs))

        # 2. Process local file paths if any
        if local_paths:
            read_tasks.extend(
                _process_local_files(
                    local_paths,
                    allowed_suffix,
                    kv_backend,
                    working_dir,
                    parallelism,
                    recursive,
                    reader_kwargs,
                )
            )

        # 3. Validate we have at least one dataset
        if not read_tasks:
            raise ValueError("No datasets created from the provided input paths.")

        # 4. Combine and process datasets
        return _combine_datasets(read_tasks, read_nums, read_storage, input_path)

    except Exception as e:
        logger.error("[READ] Failed to read data from %s: %s", input_path, e)
        raise
