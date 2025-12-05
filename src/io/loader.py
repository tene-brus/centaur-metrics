"""Data loading utilities for JSONL files."""

import os

import polars as pl


class DataLoader:
    """
    Loads annotation data from JSONL files.

    Handles filtering of invalid rows and extraction of annotator columns.
    """

    def __init__(self, data_path: str, infer_schema_length: int = 8000):
        """
        Initialize loader with path to JSONL file.

        Args:
            data_path: Path to the JSONL file
            infer_schema_length: Number of rows to use for schema inference
        """
        self.data_path = data_path
        self.infer_schema_length = infer_schema_length
        self._data: pl.DataFrame | None = None
        self._annotators: list[str] | None = None

    def load(self) -> pl.DataFrame:
        """
        Load and filter the data.

        Filters out:
        - Rows with num_annotations == 0
        - Rows where predictions is null

        Returns:
            Filtered DataFrame
        """
        if self._data is not None:
            return self._data

        data = pl.read_ndjson(self.data_path, infer_schema_length=self.infer_schema_length)

        # Drop id column if present
        if "id" in data.columns:
            data = data.drop(["id"])

        # Filter valid rows
        data = data.filter(pl.col("num_annotations") != 0).filter(
            pl.col("predictions").is_not_null()
        )

        self._data = data
        return data

    @property
    def annotators(self) -> list[str]:
        """
        Get list of annotator identifiers.

        Annotators are identified by:
        - Columns containing "@" (email addresses)
        - "predictions" column (model predictions)
        - "ground_truth" column (if present)
        """
        if self._annotators is not None:
            return self._annotators

        if self._data is None:
            self.load()

        # Email columns + special columns
        email_cols = [col for col in self._data.columns if "@" in col]
        special_cols = ["predictions", "ground_truth"]

        self._annotators = email_cols + [col for col in special_cols if col in self._data.columns]
        return self._annotators

    @property
    def traders(self) -> list[str]:
        """Get unique trader names from the data."""
        if self._data is None:
            self.load()

        if "trader" not in self._data.columns:
            return []

        return self._data["trader"].unique().to_list()

    def filter_by_trader(self, trader: str) -> pl.DataFrame:
        """Get data for a specific trader."""
        if self._data is None:
            self.load()

        return self._data.filter(pl.col("trader") == trader)

    @property
    def base_name(self) -> str:
        """Get base name of the data file (without extension)."""
        return os.path.splitext(os.path.basename(self.data_path))[0]
