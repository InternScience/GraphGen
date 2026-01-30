from typing import TYPE_CHECKING, List, Union

from graphgen.bases.base_reader import BaseReader

if TYPE_CHECKING:
    import ray
    from ray.data import Dataset


class TXTReader(BaseReader):
    def read(
        self,
        input_path: Union[str, List[str]],
    ) -> "Dataset":
        """
        Read text files from the specified input path.
        :param input_path: Path to the input text file or list of text files.
        :return: Ray Dataset containing the read text data.
        """
        import ray

        docs_ds = ray.data.read_binary_files(
            input_path,
            include_paths=True,
        )

        docs_ds = docs_ds.map(
            lambda row: {
                "type": "text",
                self.text_column: row["bytes"].decode("utf-8"),
                "path": row["path"],
            }
        )

        docs_ds = docs_ds.filter(self._should_keep_item)
        return docs_ds
