import os
from dataclasses import dataclass
    def iter_items(self) -> Iterator[Tuple[str, dict]]:
        """
        Iterate over all items without loading everything into memory at once.
        Returns an iterator of (key, value) tuples.
        """
        for key, value in self._data.items():
            yield key, value

    def get_batch(self, keys: list[str]) -> dict[str, dict]:
        """
        Get a batch of items by their keys.

        :param keys: List of keys to retrieve.
        :return: Dictionary of {key: value} for the requested keys.
        """
        return {key: self._data.get(key) for key in keys if key in self._data}

    def iter_batches(self, batch_size: int = 10000) -> Iterator[dict[str, dict]]:
        """
        Iterate over items in batches to avoid loading everything into memory.

        :param batch_size: Number of items per batch.
        :return: Iterator of dictionaries, each containing up to batch_size items.
        """
        batch = {}
        count = 0
        for key, value in self._data.items():
            batch[key] = value
            count += 1
            if count >= batch_size:
                yield batch
                batch = {}
                count = 0
        if batch:
            yield batch

    def filter_keys(self, data: list[str]) -> set[str]:
        return {s for s in data if s not in self._data}

    def upsert(self, data: dict):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        if left_data:
            self._data.update(left_data)
        return left_data

    def drop(self):
        if self._data:
            self._data.clear()

    def reload(self):
        self._data = load_json(self._file_name) or {}
        print(f"Reload KV {self.namespace} with {len(self._data)} data")
