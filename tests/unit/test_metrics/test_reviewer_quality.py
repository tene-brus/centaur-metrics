"""Tests for src/metrics/reviewer_quality.py."""

import json
from pathlib import Path

import polars as pl
import pytest

from src.metrics.reviewer_quality import (
    ReviewerErrorFrequency,
    annotations_match,
    calculate_reviewer_error_frequency,
    calculate_reviewer_error_frequency_from_file,
)


class TestAnnotationsMatch:
    """Tests for annotations_match function."""

    def test_returns_true_for_identical_annotations(self):
        """Should return True when annotations are identical."""
        trades_a = [
            {
                "asset_reference_type": "Majors",
                "direction": "Long",
                "exposure_change": "Increase",
                "position_status": "Clearly a new position",
            }
        ]
        trades_b = [
            {
                "asset_reference_type": "Majors",
                "direction": "Long",
                "exposure_change": "Increase",
                "position_status": "Clearly a new position",
            }
        ]

        assert annotations_match(trades_a, trades_b) is True

    def test_returns_false_for_different_direction(self):
        """Should return False when direction differs."""
        trades_a = [{"direction": "Long", "exposure_change": "Increase"}]
        trades_b = [{"direction": "Short", "exposure_change": "Increase"}]

        assert annotations_match(trades_a, trades_b) is False

    def test_returns_false_for_different_exposure_change(self):
        """Should return False when exposure_change differs."""
        trades_a = [{"direction": "Long", "exposure_change": "Increase"}]
        trades_b = [{"direction": "Long", "exposure_change": "Decrease"}]

        assert annotations_match(trades_a, trades_b) is False

    def test_returns_false_for_different_lengths(self):
        """Should return False when trade counts differ."""
        trades_a = [
            {"direction": "Long"},
            {"direction": "Short"},
        ]
        trades_b = [{"direction": "Long"}]

        assert annotations_match(trades_a, trades_b) is False

    def test_returns_true_for_empty_lists(self):
        """Should return True for two empty lists."""
        assert annotations_match([], []) is True

    def test_handles_different_order(self):
        """Should match trades regardless of order."""
        trades_a = [
            {"direction": "Long", "asset_reference_type": "Majors"},
            {"direction": "Short", "asset_reference_type": "DeFi"},
        ]
        trades_b = [
            {"direction": "Short", "asset_reference_type": "DeFi"},
            {"direction": "Long", "asset_reference_type": "Majors"},
        ]

        assert annotations_match(trades_a, trades_b) is True

    def test_handles_specific_assets(self):
        """Should compare specific_assets correctly."""
        trades_a = [{"specific_assets": ["BTC", "ETH"]}]
        trades_b = [{"specific_assets": ["BTC", "ETH"]}]

        assert annotations_match(trades_a, trades_b) is True

    def test_handles_none_specific_assets(self):
        """Should handle None specific_assets."""
        trades_a = [{"specific_assets": None, "direction": "Long"}]
        trades_b = [{"specific_assets": None, "direction": "Long"}]

        assert annotations_match(trades_a, trades_b) is True


class TestCalculateReviewerErrorFrequency:
    """Tests for calculate_reviewer_error_frequency function."""

    @pytest.fixture
    def sample_data_all_match(self):
        """Data where reviewer matches GT on all tasks."""
        return pl.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "trader": ["trader_A", "trader_A"],
                "reviewer@example.com": [
                    [
                        {
                            "label_type": "action",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                        }
                    ],
                    [
                        {
                            "label_type": "action",
                            "direction": "Short",
                            "action_exposure_change": "Decrease",
                        }
                    ],
                ],
                "ground_truth": [
                    [
                        {
                            "label_type": "action",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                        }
                    ],
                    [
                        {
                            "label_type": "action",
                            "direction": "Short",
                            "action_exposure_change": "Decrease",
                        }
                    ],
                ],
            }
        )

    @pytest.fixture
    def sample_data_some_errors(self):
        """Data where reviewer has some errors."""
        return pl.DataFrame(
            {
                "task_id": ["t1", "t2", "t3", "t4"],
                "trader": ["trader_A", "trader_A", "trader_B", "trader_B"],
                "reviewer@example.com": [
                    [{"label_type": "action", "direction": "Long"}],  # Match
                    [
                        {"label_type": "action", "direction": "Short"}
                    ],  # Error - different from GT
                    [{"label_type": "action", "direction": "Long"}],  # Match
                    [
                        {"label_type": "action", "direction": "Long"}
                    ],  # Error - different from GT
                ],
                "ground_truth": [
                    [{"label_type": "action", "direction": "Long"}],
                    [
                        {"label_type": "action", "direction": "Long"}
                    ],  # GT says Long, reviewer said Short
                    [{"label_type": "action", "direction": "Long"}],
                    [
                        {"label_type": "action", "direction": "Short"}
                    ],  # GT says Short, reviewer said Long
                ],
            }
        )

    def test_returns_none_if_reviewer_not_in_columns(self):
        """Should return None if reviewer email not found."""
        data = pl.DataFrame({"task_id": ["t1"], "ground_truth": [[]]})

        result = calculate_reviewer_error_frequency(data, "missing@example.com")

        assert result is None

    def test_returns_none_if_no_ground_truth_column(self):
        """Should return None if ground_truth column missing."""
        data = pl.DataFrame({"task_id": ["t1"], "reviewer@example.com": [[]]})

        result = calculate_reviewer_error_frequency(data, "reviewer@example.com")

        assert result is None

    def test_returns_result_with_zero_reviewed_if_no_common_rows(self):
        """Should return result with zero reviewed tasks if no rows have both reviewer and GT."""
        data = pl.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "reviewer@example.com": [[{"direction": "Long"}], None],
                "ground_truth": [None, [{"direction": "Long"}]],
            }
        )

        result = calculate_reviewer_error_frequency(data, "reviewer@example.com")

        # Now returns result even with no reviewed tasks, showing non-reviewed stats
        assert result is not None
        assert result.total_tasks == 0
        assert result.tasks_not_reviewed == 1
        assert result.error_frequency == 0.0

    def test_returns_zero_error_frequency_when_all_match(self, sample_data_all_match):
        """Should return 0 error frequency when all annotations match GT."""
        result = calculate_reviewer_error_frequency(
            sample_data_all_match, "reviewer@example.com", "test_project"
        )

        assert result is not None
        assert result.total_tasks == 2
        assert result.tasks_with_errors == 0
        assert result.error_frequency == 0.0

    def test_calculates_error_frequency_correctly(self, sample_data_some_errors):
        """Should calculate error frequency as errors / total."""
        result = calculate_reviewer_error_frequency(
            sample_data_some_errors, "reviewer@example.com", "test_project"
        )

        assert result is not None
        assert result.total_tasks == 4
        assert result.tasks_with_errors == 2
        assert result.error_frequency == 0.5  # 2 / 4

    def test_includes_project_name(self, sample_data_all_match):
        """Should include project name in result."""
        result = calculate_reviewer_error_frequency(
            sample_data_all_match, "reviewer@example.com", "my_project"
        )

        assert result.project_name == "my_project"

    def test_includes_reviewer_email(self, sample_data_all_match):
        """Should include reviewer email in result."""
        result = calculate_reviewer_error_frequency(
            sample_data_all_match, "reviewer@example.com", "test_project"
        )

        assert result.reviewer_email == "reviewer@example.com"

    def test_calculates_per_trader_breakdown(self, sample_data_some_errors):
        """Should calculate error frequency per trader."""
        result = calculate_reviewer_error_frequency(
            sample_data_some_errors, "reviewer@example.com"
        )

        assert "trader_A" in result.per_trader
        assert "trader_B" in result.per_trader

        # trader_A: 2 tasks, 1 error
        assert result.per_trader["trader_A"]["total"] == 2
        assert result.per_trader["trader_A"]["errors"] == 1
        assert result.per_trader["trader_A"]["frequency"] == 0.5

        # trader_B: 2 tasks, 1 error
        assert result.per_trader["trader_B"]["total"] == 2
        assert result.per_trader["trader_B"]["errors"] == 1
        assert result.per_trader["trader_B"]["frequency"] == 0.5

    def test_handles_missing_trader_column(self):
        """Should use 'Unknown' when trader column is missing."""
        data = pl.DataFrame(
            {
                "task_id": ["t1"],
                "reviewer@example.com": [
                    [{"label_type": "action", "direction": "Long"}]
                ],
                "ground_truth": [[{"label_type": "action", "direction": "Long"}]],
            }
        )

        result = calculate_reviewer_error_frequency(data, "reviewer@example.com")

        assert result is not None
        assert "Unknown" in result.per_trader

    def test_returns_dataclass_instance(self, sample_data_all_match):
        """Should return ReviewerErrorFrequency dataclass."""
        result = calculate_reviewer_error_frequency(
            sample_data_all_match, "reviewer@example.com"
        )

        assert isinstance(result, ReviewerErrorFrequency)


class TestCalculateReviewerErrorFrequencyFromFile:
    """Tests for calculate_reviewer_error_frequency_from_file function."""

    def test_loads_and_calculates_from_file(self, tmp_path):
        """Should load JSONL file and calculate error frequency."""
        jsonl_path = tmp_path / "test_project.jsonl"
        content = [
            {
                "task_id": "t1",
                "trader": "A",
                "reviewer@example.com": [{"label_type": "action", "direction": "Long"}],
                "ground_truth": [{"label_type": "action", "direction": "Long"}],
            },
            {
                "task_id": "t2",
                "trader": "A",
                "reviewer@example.com": [
                    {"label_type": "action", "direction": "Short"}
                ],
                "ground_truth": [{"label_type": "action", "direction": "Long"}],
            },
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        result = calculate_reviewer_error_frequency_from_file(
            str(jsonl_path), "reviewer@example.com"
        )

        assert result is not None
        assert result.project_name == "test_project"
        assert result.total_tasks == 2
        assert result.tasks_with_errors == 1
        assert result.error_frequency == 0.5

    def test_extracts_project_name_from_filename(self, tmp_path):
        """Should extract project name from file basename."""
        jsonl_path = tmp_path / "my_custom_project.jsonl"
        content = [
            {
                "task_id": "t1",
                "trader": "A",
                "reviewer@example.com": [{"direction": "Long"}],
                "ground_truth": [{"direction": "Long"}],
            },
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        result = calculate_reviewer_error_frequency_from_file(
            str(jsonl_path), "reviewer@example.com"
        )

        assert result.project_name == "my_custom_project"

    def test_returns_none_for_missing_reviewer(self, tmp_path):
        """Should return None if reviewer not in data."""
        jsonl_path = tmp_path / "test.jsonl"
        content = [
            {
                "task_id": "t1",
                "other@example.com": [{"direction": "Long"}],
                "ground_truth": [{"direction": "Long"}],
            },
        ]
        with open(jsonl_path, "w") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")

        result = calculate_reviewer_error_frequency_from_file(
            str(jsonl_path), "reviewer@example.com"
        )

        assert result is None
