import json
import os
from typing import Any, Dict, Iterator, List, Union

import ray
import ray.data

from graphgen.bases.base_reader import BaseReader
from graphgen.utils import logger


class JSONReader(BaseReader):
    """
    Reader for JSON and JSONL files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(self, input_path: Union[str, List[str]]) -> ray.data.Dataset:
        """
        Read JSON file and return Ray Dataset.
        :param input_path: Path to JSON/JSONL file or list of JSON/JSONL files.
        :return: Ray Dataset containing validated and filtered data.
        """
        if self.modalities and len(self.modalities) >= 2:
            ds: ray.data.Dataset = ray.data.from_items([])
            for file in input_path if isinstance(input_path, list) else [input_path]:
                data = []
                if file.endswith(".jsonl"):
                    with open(file, "r", encoding="utf-8") as f:
                        for line in f:
                            item = json.loads(line)
                            data.append(item)
                else:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data = self._unify_schema(data)
                file_ds: ray.data.Dataset = ray.data.from_items(data)
                ds = ds.union(file_ds)  # type: ignore
        else:
            ds = ray.data.read_json(input_path)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds

    def read_stream(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream read JSONL files line by line without loading entire file into memory.
        Returns an iterator that yields filtered documents.

        :param file_path: Path to the JSONL file.
        :return: Iterator of dictionaries containing the data.
        """
        if not file_path.endswith(".jsonl"):
            raise ValueError("read_stream only supports JSONL files, not JSON files")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    assert "type" in doc, f"Missing 'type' in document: {doc}"
                    if doc.get("type") == "text" and self.text_column not in doc:
                        raise ValueError(
                            f"Missing '{self.text_column}' in document: {doc}"
                        )

                    # Apply filtering logic inline (similar to BaseReader.filter)
                    if doc.get("type") == "text":
                        content = doc.get(self.text_column, "").strip()
                        if content:
                            yield doc
                    elif doc.get("type") in ("image", "table", "equation"):
                        img_path = doc.get("img_path")
                        if self._image_exists(img_path):
                            yield doc
                    else:
                        yield doc
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON line: %s. Error: %s", line, e)

    @staticmethod
    def _image_exists(path_or_url: str, timeout: int = 3) -> bool:
        """
        Check if an image exists at the given local path or URL.
        :param path_or_url: Local file path or remote URL of the image.
        :param timeout: Timeout for remote URL requests in seconds.
        :return: True if the image exists, False otherwise.
        """
        if not path_or_url:
            return False
        if not path_or_url.startswith(("http://", "https://", "ftp://")):
            path = path_or_url.replace("file://", "", 1)
            path = os.path.abspath(path)
            return os.path.isfile(path)
        try:
            import requests
            resp = requests.head(path_or_url, allow_redirects=True, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _unify_schema(data):
        """
        Unify schema for JSON data.
        """
        for item in data:
            if "content" in item and isinstance(item["content"], dict):
                item["content"] = json.dumps(item["content"])
        return data
