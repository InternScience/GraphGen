import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests


class BaseReader(ABC):
    """
    Abstract base class for reading and processing data.
    """

    def __init__(self, text_column: str = "content"):
        self.text_column = text_column

    @abstractmethod
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read data from the specified file path.

        :param file_path: Path to the input file.
        :return: List of dictionaries containing the data.
        """

    @staticmethod
    def filter(data: List[dict]) -> List[dict]:
        """
        Filter out entries with empty or missing text in the specified column.

        :param data: List of dictionaries containing the data.
        :return: Filtered list of dictionaries.
        """

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
                resp = requests.head(path_or_url, allow_redirects=True, timeout=timeout)
                return resp.status_code == 200
            except requests.RequestException:
                return False

        filtered_data = []
        for item in data:
            if item.get("type") == "text":
                content = item.get("content", "").strip()
                if content:
                    filtered_data.append(item)
            elif item.get("type") in ("image", "table", "equation"):
                img_path = item.get("img_path")
                if _image_exists(img_path):
                    filtered_data.append(item)
            else:
                filtered_data.append(item)
        return filtered_data
