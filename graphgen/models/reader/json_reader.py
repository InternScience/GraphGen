import json
import os
from typing import Any, Dict, List, Union

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
