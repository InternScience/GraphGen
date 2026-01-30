from typing import TYPE_CHECKING, List, Union

from graphgen.bases.base_reader import BaseReader

if TYPE_CHECKING:
    import ray
    from ray.data import Dataset


class ParquetReader(BaseReader):
    """
    Read parquet files, requiring the schema to be restored to List[Dict[str, Any]].
    Columns:
    - type: The type of the document (e.g., "text", "image", etc.)
    - if type is "text", "content" column must be present.
    """

    def read(self, input_path: Union[str, List[str]]) -> "Dataset":
        """
        Read Parquet files using Ray Data.

        :param input_path: Path to Parquet file or list of Parquet files.
        :return: Ray Dataset containing validated documents.
        """
        import ray

        if not ray.is_initialized():
            ray.init()

        ds = ray.data.read_parquet(input_path, include_paths=True)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds
