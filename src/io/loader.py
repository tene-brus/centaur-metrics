"""Data loading utilities for JSONL files."""

import json
import os

import polars as pl

# Default config path (relative to app/data/)
DEFAULT_REVIEWER_CONFIG = "reviewer_config.json"


def load_reviewer_config(config_path: str | None = None) -> dict:
    """Load reviewer configuration from JSON file.

    Args:
        config_path: Path to config file. If None, searches in common locations.

    Returns:
        Config dict with 'global_exclusions' and 'project_reviewers' keys.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)

    # Search in common locations
    search_paths = [
        DEFAULT_REVIEWER_CONFIG,
        os.path.join("app", "data", DEFAULT_REVIEWER_CONFIG),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "app",
            "data",
            DEFAULT_REVIEWER_CONFIG,
        ),
    ]

    for path in search_paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    # Return empty config if not found
    return {"global_exclusions": [], "project_reviewers": {}}


def get_excluded_annotators(
    project_name: str | None = None, config_path: str | None = None
) -> list[str]:
    """Get list of annotators to exclude for a given project.

    Args:
        project_name: Name of the project (without _metrics suffix)
        config_path: Optional path to reviewer config file

    Returns:
        List of annotator emails to exclude
    """
    config = load_reviewer_config(config_path)

    # Start with global exclusions
    excluded = list(config.get("global_exclusions", []))

    # Add project-specific reviewers
    if project_name:
        # Strip _metrics suffix if present
        clean_name = project_name.replace("_metrics", "")
        project_reviewers = config.get("project_reviewers", {}).get(clean_name, [])
        excluded.extend(project_reviewers)

    return list(set(excluded))  # Remove duplicates


class DataLoader:
    """
    Loads annotation data from JSONL files.

    Handles filtering of invalid rows and extraction of annotator columns.
    """

    def __init__(
        self,
        data_path: str,
        infer_schema_length: int = 8000,
        config_path: str | None = None,
    ):
        """
        Initialize loader with path to JSONL file.

        Args:
            data_path: Path to the JSONL file
            infer_schema_length: Number of rows to use for schema inference
            config_path: Optional path to reviewer config file
        """
        self.data_path = data_path
        self.infer_schema_length = infer_schema_length
        self.config_path = config_path
        self._data: pl.DataFrame | None = None
        self._annotators: list[str] | None = None
        self._excluded_annotators: list[str] | None = None

    def load(self) -> pl.DataFrame:
        """
        Load and filter the data.

        Filters out:
        - Rows with num_annotations == 0

        Returns:
            Filtered DataFrame
        """
        if self._data is not None:
            return self._data

        data = pl.read_ndjson(
            self.data_path, infer_schema_length=self.infer_schema_length
        )

        # Drop id column if present
        if "id" in data.columns:
            data = data.drop(["id"])

        # Filter valid rows
        data = data.filter(pl.col("num_annotations") != 0)

        self._data = data
        return data

    @property
    def excluded_annotators(self) -> list[str]:
        """Get list of annotators to exclude based on config."""
        if self._excluded_annotators is None:
            self._excluded_annotators = get_excluded_annotators(
                project_name=self.base_name, config_path=self.config_path
            )
        return self._excluded_annotators

    @property
    def annotators(self) -> list[str]:
        """
        Get list of annotator identifiers.

        Annotators are identified by:
        - Columns containing "@" (email addresses)
        - "predictions" column (model predictions)
        - "ground_truth" column (if present)

        Excludes annotators based on reviewer_config.json (global + project-specific).
        """
        if self._annotators is not None:
            return self._annotators

        if self._data is None:
            self.load()

        # Email columns + special columns, excluding configured annotators
        excluded = self.excluded_annotators
        email_cols = [
            col for col in self._data.columns if "@" in col and col not in excluded
        ]
        special_cols = ["predictions", "ground_truth"]

        self._annotators = email_cols + [
            col for col in special_cols if col in self._data.columns
        ]
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
