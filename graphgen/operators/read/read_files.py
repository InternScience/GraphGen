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


def read_files(file_path: str, cache_dir: str | None = None) -> list[dict]:
    suffix = file_path.split(".")[-1].lower()
    if suffix == "pdf":
        if cache_dir is not None:
            reader = _MAPPING[suffix](output_dir=cache_dir)
        else:
            reader = _MAPPING[suffix]()
    elif suffix in _MAPPING:
        reader = _MAPPING[suffix]()
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats are: {list(_MAPPING.keys())}"
        )
    return reader.read(file_path)
