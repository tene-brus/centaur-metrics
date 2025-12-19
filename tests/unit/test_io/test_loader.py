"""Tests for src/io/loader.py."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.io.loader import (
    DataLoader,
    load_reviewer_config,
    get_excluded_annotators,
)


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_returns_dataframe(self, temp_jsonl_file):
        """Should return a polars DataFrame."""
        loader = DataLoader(temp_jsonl_file)
        data = loader.load()

        assert isinstance(data, pl.DataFrame)

    def test_load_filters_zero_annotations(self, temp_jsonl_file):
        """Should filter out rows with num_annotations == 0."""
        loader = DataLoader(temp_jsonl_file)
        data = loader.load()

        # task_4 has num_annotations=0 and should be filtered
        assert data.filter(pl.col("num_annotations") == 0).shape[0] == 0

    def test_load_filters_null_predictions(self, tmp_path):
        """Should filter out rows with null predictions."""
        jsonl_path = tmp_path / "test.jsonl"
        content = [
            {
                "task_id": "1",
                "trader": "A",
                "num_annotations": 1,
                "predictions": None,
                "user@example.com": [{"direction": "Long"}],
            },
            {
                "task_id": "2",
                "trader": "A",
                "num_annotations": 1,
                "predictions": [{"direction": "Long"}],
                "user@example.com": [{"direction": "Long"}],
            },
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        loader = DataLoader(str(jsonl_path))
        data = loader.load()

        # Only task_2 should remain
        assert data.shape[0] == 1

    def test_load_drops_id_column(self, tmp_path):
        """Should drop 'id' column if present."""
        jsonl_path = tmp_path / "test.jsonl"
        content = [
            {
                "id": 123,
                "task_id": "1",
                "trader": "A",
                "num_annotations": 1,
                "predictions": [{"direction": "Long"}],
                "user@example.com": [{"direction": "Long"}],
            }
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        loader = DataLoader(str(jsonl_path))
        data = loader.load()

        assert "id" not in data.columns

    def test_load_caches_result(self, temp_jsonl_file):
        """Should cache loaded data on subsequent calls."""
        loader = DataLoader(temp_jsonl_file)
        data1 = loader.load()
        data2 = loader.load()

        # Should be the same object (cached)
        assert data1 is data2

    def test_annotators_property(self, temp_jsonl_file):
        """Should return list of annotator identifiers."""
        loader = DataLoader(temp_jsonl_file)
        annotators = loader.annotators

        assert "user1@example.com" in annotators
        assert "user2@example.com" in annotators
        assert "predictions" in annotators

    def test_annotators_includes_ground_truth(self, temp_jsonl_file):
        """Should include ground_truth in annotators if present."""
        loader = DataLoader(temp_jsonl_file)
        annotators = loader.annotators

        assert "ground_truth" in annotators

    def test_annotators_excludes_configured_annotators(self, tmp_path):
        """Should exclude annotators based on reviewer_config.json."""
        # Create a test config file
        config_path = tmp_path / "reviewer_config.json"
        config = {
            "global_exclusions": ["excluded@example.com"],
            "project_reviewers": {"test": ["project_reviewer@example.com"]},
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create test JSONL
        jsonl_path = tmp_path / "test.jsonl"
        content = [
            {
                "task_id": "1",
                "trader": "A",
                "num_annotations": 1,
                "predictions": [{"direction": "Long"}],
                "excluded@example.com": [{"direction": "Long"}],  # Global exclusion
                "project_reviewer@example.com": [{"direction": "Long"}],  # Project reviewer
                "valid@example.com": [{"direction": "Long"}],
            }
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        loader = DataLoader(str(jsonl_path), config_path=str(config_path))
        annotators = loader.annotators

        assert "excluded@example.com" not in annotators
        assert "project_reviewer@example.com" not in annotators
        assert "valid@example.com" in annotators

    def test_traders_property(self, temp_jsonl_file):
        """Should return unique trader names."""
        loader = DataLoader(temp_jsonl_file)
        traders = loader.traders

        assert "trader_A" in traders
        assert "trader_B" in traders
        assert len(traders) == 2

    def test_traders_empty_when_no_column(self, tmp_path):
        """Should return empty list if no trader column."""
        jsonl_path = tmp_path / "test.jsonl"
        content = [
            {
                "task_id": "1",
                "num_annotations": 1,
                "predictions": [{"direction": "Long"}],
                "user@example.com": [{"direction": "Long"}],
            }
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        loader = DataLoader(str(jsonl_path))
        traders = loader.traders

        assert traders == []

    def test_filter_by_trader(self, temp_jsonl_file):
        """Should filter data by specific trader."""
        loader = DataLoader(temp_jsonl_file)
        trader_a_data = loader.filter_by_trader("trader_A")

        # All rows should have trader_A
        traders = trader_a_data["trader"].to_list()
        assert all(t == "trader_A" for t in traders)

    def test_base_name_property(self, temp_jsonl_file):
        """Should return base name of the data file."""
        loader = DataLoader(temp_jsonl_file)

        assert loader.base_name == "test_data"


class TestReviewerConfig:
    """Tests for reviewer config functions."""

    def test_load_reviewer_config_returns_dict(self, tmp_path):
        """load_reviewer_config should return a dict."""
        config_path = tmp_path / "reviewer_config.json"
        config = {
            "global_exclusions": ["test@example.com"],
            "project_reviewers": {},
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = load_reviewer_config(str(config_path))

        assert isinstance(result, dict)
        assert "global_exclusions" in result

    def test_load_reviewer_config_uses_specified_path(self, tmp_path):
        """load_reviewer_config should use specified path when provided and exists."""
        config_path = tmp_path / "custom_config.json"
        config = {
            "global_exclusions": ["custom@example.com"],
            "project_reviewers": {},
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = load_reviewer_config(str(config_path))

        assert result["global_exclusions"] == ["custom@example.com"]

    def test_get_excluded_annotators_returns_global_exclusions(self, tmp_path):
        """get_excluded_annotators should return global exclusions."""
        config_path = tmp_path / "reviewer_config.json"
        config = {
            "global_exclusions": ["global@example.com"],
            "project_reviewers": {},
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = get_excluded_annotators(config_path=str(config_path))

        assert "global@example.com" in result

    def test_get_excluded_annotators_includes_project_reviewers(self, tmp_path):
        """get_excluded_annotators should include project-specific reviewers."""
        config_path = tmp_path / "reviewer_config.json"
        config = {
            "global_exclusions": ["global@example.com"],
            "project_reviewers": {
                "my_project": ["reviewer@example.com"],
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = get_excluded_annotators(
            project_name="my_project", config_path=str(config_path)
        )

        assert "global@example.com" in result
        assert "reviewer@example.com" in result

    def test_get_excluded_annotators_strips_metrics_suffix(self, tmp_path):
        """get_excluded_annotators should strip _metrics suffix from project name."""
        config_path = tmp_path / "reviewer_config.json"
        config = {
            "global_exclusions": [],
            "project_reviewers": {
                "my_project": ["reviewer@example.com"],
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = get_excluded_annotators(
            project_name="my_project_metrics", config_path=str(config_path)
        )

        assert "reviewer@example.com" in result

    def test_get_excluded_annotators_removes_duplicates(self, tmp_path):
        """get_excluded_annotators should remove duplicate emails."""
        config_path = tmp_path / "reviewer_config.json"
        config = {
            "global_exclusions": ["same@example.com"],
            "project_reviewers": {
                "my_project": ["same@example.com"],  # Duplicate
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = get_excluded_annotators(
            project_name="my_project", config_path=str(config_path)
        )

        # Should only appear once
        assert result.count("same@example.com") == 1
