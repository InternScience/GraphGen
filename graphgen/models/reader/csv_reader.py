from typing import TYPE_CHECKING, List, Union

from graphgen.bases.base_reader import BaseReader

if TYPE_CHECKING:
    import ray
    from ray.data import Dataset


class CSVReader(BaseReader):
    """
    Reader for CSV files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(self, input_path: Union[str, List[str]]) -> "Dataset":
        """
        Read CSV files and return Ray Dataset.

        :param input_path: Path to CSV file or list of CSV files.
        :return: Ray Dataset containing validated and filtered data.
        """
        import ray

        ds = ray.data.read_csv(input_path, include_paths=True)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds
