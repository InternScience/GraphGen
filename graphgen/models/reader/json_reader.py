import json
from typing import TYPE_CHECKING, List, Union

from graphgen.bases.base_reader import BaseReader

if TYPE_CHECKING:
    import ray
    import ray.data


class JSONReader(BaseReader):
    """
    Reader for JSON and JSONL files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(self, input_path: Union[str, List[str]]) -> "ray.data.Dataset":
        """
        Read JSON file and return Ray Dataset.
        :param input_path: Path to JSON/JSONL file or list of JSON/JSONL files.
        :return: Ray Dataset containing validated and filtered data.
        """
        import ray

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
                # add path
                for item in data:
                    item["path"] = file
                file_ds: ray.data.Dataset = ray.data.from_items(data)
                ds = ds.union(file_ds)  # type: ignore
        else:
            ds = ray.data.read_json(input_path, include_paths=True)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds

    @staticmethod
    def _unify_schema(data):
        """
        Unify schema for JSON data.
        """
        for item in data:
            if "content" in item and isinstance(item["content"], dict):
                item["content"] = json.dumps(item["content"])
        return data
