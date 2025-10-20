from graphgen.models import CSVReader, JSONLReader, JSONReader, PDFReader, TXTReader

_MAPPING = {
    "jsonl": JSONLReader,
    "json": JSONReader,
    "txt": TXTReader,
    "csv": CSVReader,
    "pdf": PDFReader,
}


def read_files(file_path: str, cache_dir: str):
    suffix = file_path.split(".")[-1].lower()
    if suffix == "pdf":
        reader = _MAPPING[suffix](output_dir=cache_dir)
    elif suffix in _MAPPING:
        reader = _MAPPING[suffix]()
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats are: {list(_MAPPING.keys())}"
        )
    return reader.read(file_path)
