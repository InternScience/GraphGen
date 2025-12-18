import json
import os
from typing import Any, Dict, Iterator, List, Union

import ray
import ray.data

from graphgen.bases.base_reader import BaseReader
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
