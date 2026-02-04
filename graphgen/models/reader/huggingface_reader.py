"""
Hugging Face Datasets Reader
This module provides a reader for accessing datasets from Hugging Face Hub.
"""

from typing import TYPE_CHECKING, List, Optional, Union

from graphgen.bases.base_reader import BaseReader

if TYPE_CHECKING:
    import numpy as np
    import ray
    from ray.data import Dataset


class HuggingFaceReader(BaseReader):
    """
    Reader for Hugging Face Datasets.

    Supports loading datasets from the Hugging Face Hub.
    Can specify a dataset by name and optional subset/split.

    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present (or specify via text_column).

    Example:
        reader = HuggingFaceReader(text_column="text")
        ds = reader.read("wikitext")
        # or with split and subset
        ds = reader.read("wikitext:wikitext-103-v1:train")
    """

    def __init__(
        self,
        text_column: str = "content",
        modalities: Optional[list] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize HuggingFaceReader.

        :param text_column: Column name containing text content
        :param modalities: List of supported modalities
        :param cache_dir: Directory to cache downloaded datasets
        :param trust_remote_code: Whether to trust remote code in datasets
        """
        super().__init__(text_column=text_column, modalities=modalities)
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

    def read(
        self,
        input_path: Union[str, List[str]],
        split: Optional[str] = None,
        subset: Optional[str] = None,
        streaming: bool = False,
        limit: Optional[int] = None,
    ) -> "Dataset":
        """
        Read dataset from Hugging Face Hub.

        :param input_path: Dataset identifier(s) from Hugging Face Hub
                          Format: "dataset_name" or "dataset_name:subset:split"
                          Example: "wikitext" or "wikitext:wikitext-103-v1:train"
        :param split: Specific split to load (overrides split in path)
        :param subset: Specific subset/configuration to load (overrides subset in path)
        :param streaming: Whether to stream the dataset instead of downloading
        :param limit: Maximum number of samples to load
        :return: Ray Dataset containing the data
        """
        try:
            import datasets as hf_datasets
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to use HuggingFaceReader. "
                "Please install it with: pip install datasets"
            ) from exc

        if isinstance(input_path, list):
            # Handle multiple datasets
            all_dss = []
            for path in input_path:
                ds = self._load_single_dataset(
                    path,
                    split=split,
                    subset=subset,
                    streaming=streaming,
                    limit=limit,
                    hf_datasets=hf_datasets,
                )
                all_dss.append(ds)

            if len(all_dss) == 1:
                combined_ds = all_dss[0]
            else:
                combined_ds = all_dss[0].union(*all_dss[1:])
        else:
            combined_ds = self._load_single_dataset(
                input_path,
                split=split,
                subset=subset,
                streaming=streaming,
                limit=limit,
                hf_datasets=hf_datasets,
            )

        # Validate and filter
        combined_ds = combined_ds.map_batches(
            self._validate_batch, batch_format="pandas"
        )
        combined_ds = combined_ds.filter(self._should_keep_item)

        return combined_ds

    def _load_single_dataset(
        self,
        dataset_path: str,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        streaming: bool = False,
        limit: Optional[int] = None,
        hf_datasets=None,
    ) -> "Dataset":
        """
        Load a single dataset from Hugging Face Hub.

        :param dataset_path: Dataset path, can include subset and split
        :param split: Override split
        :param subset: Override subset
        :param streaming: Whether to stream
        :param limit: Max samples
        :param hf_datasets: Imported datasets module
        :return: Ray Dataset
        """
        import numpy as np
        import ray

        # Parse dataset path format: "dataset_name:subset:split"
        parts = dataset_path.split(":")
        dataset_name = parts[0]
        parsed_subset = parts[1] if len(parts) > 1 else None
        parsed_split = parts[2] if len(parts) > 2 else None

        # Override with explicit parameters
        final_subset = subset or parsed_subset
        final_split = split or parsed_split or "train"

        # Load dataset from Hugging Face
        load_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "streaming": streaming,
        }

        if final_subset:
            load_kwargs["name"] = final_subset

        hf_dataset = hf_datasets.load_dataset(
            dataset_name, split=final_split, **load_kwargs
        )

        # Apply limit before converting to Ray dataset for memory efficiency
        if limit:
            if streaming:
                hf_dataset = hf_dataset.take(limit)
            else:
                hf_dataset = hf_dataset.select(range(limit))

        # Convert to Ray dataset using lazy evaluation
        ray_ds = ray.data.from_huggingface(hf_dataset)

        # Define batch processing function for lazy evaluation
        def _process_batch(batch: dict[str, "np.ndarray"]) -> dict[str, "np.ndarray"]:
            """
            Process a batch of data to add type field and rename text column.

            :param batch: A dictionary with column names as keys and numpy arrays
            :return: Processed batch dictionary with numpy arrays
            """
            if not batch:
                return {}

            # Get the number of rows in the batch
            num_rows = len(next(iter(batch.values())))

            # Add type field if not present
            if "type" not in batch:
                batch["type"] = np.array(["text"] * num_rows)

            # Rename text_column to 'content' if different
            if self.text_column != "content" and self.text_column in batch:
                batch["content"] = batch.pop(self.text_column)

            return batch

        # Apply post-processing using map_batches for distributed lazy evaluation
        ray_ds = ray_ds.map_batches(_process_batch)

        return ray_ds
