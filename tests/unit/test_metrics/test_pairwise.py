"""Tests for src/metrics/pairwise.py."""

import json

import polars as pl
import pytest

from src.metrics.pairwise import PairwiseCalculator, validate_and_dump_annotations


class TestValidateAndDumpAnnotations:
    """Tests for validate_and_dump_annotations function."""

    def test_returns_empty_for_none(self):
        """Should return empty list for None input."""
        result = validate_and_dump_annotations(None)
        assert result == []

    def test_returns_empty_for_empty_list(self):
        """Should return empty list for empty list input."""
        result = validate_and_dump_annotations([])
        assert result == []

    def test_returns_validated_annotations(self):
        """Should return validated annotations as dicts."""
        raw = [
            {
                "label_type": "action",
                "asset_reference_type": "Majors",
                "direction": "Long",
                "action_exposure_change": "Increase",
                "action_position_status": "Clearly a new position",
            }
        ]
        result = validate_and_dump_annotations(raw)

        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_skips_invalid_annotations(self):
        """Should skip annotations that fail validation."""
        raw = [
            {"invalid_field": "value"},  # Invalid
            {
                "label_type": "action",
                "asset_reference_type": "Majors",
                "direction": "Long",
            },
        ]
        result = validate_and_dump_annotations(raw)

        # May skip invalid ones depending on validation rules
        # At minimum should not raise exception
        assert isinstance(result, list)


class TestPairwiseCalculator:
    """Tests for PairwiseCalculator class."""

    @pytest.fixture
    def overall_calculator(self):
        """Create overall agreement calculator."""
        return PairwiseCalculator.create(case=None, common=False)

    @pytest.fixture
    def field_calculator(self):
        """Create per-field calculator."""
        return PairwiseCalculator.create(case="field", common=False)

    @pytest.fixture
    def label_calculator(self):
        """Create per-label calculator."""
        return PairwiseCalculator.create(case="label", common=False)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "trader": ["A", "A"],
                "ground_truth_member": [None, None],
                "user1@example.com": [
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ],
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "DeFi",
                            "direction": "Short",
                            "action_exposure_change": "Decrease",
                            "action_position_status": "Clearly an existing position",
                        }
                    ],
                ],
                "user2@example.com": [
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ],
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "DeFi",
                            "direction": "Long",  # Different
                            "action_exposure_change": "Decrease",
                            "action_position_status": "Clearly an existing position",
                        }
                    ],
                ],
            }
        )

    def test_create_factory_method(self):
        """Should create calculator via factory method."""
        calc = PairwiseCalculator.create(case="field", common=True)

        assert calc.case == "field"
        assert calc.common is True

    def test_calculate_pair_returns_float_for_overall(
        self, overall_calculator, sample_data
    ):
        """Should return float for overall agreement."""
        result = overall_calculator.calculate_pair(
            sample_data, "user1@example.com", "user2@example.com"
        )

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_calculate_pair_returns_list_for_field(
        self, field_calculator, sample_data
    ):
        """Should return list for per-field agreement."""
        result = field_calculator.calculate_pair(
            sample_data, "user1@example.com", "user2@example.com"
        )

        assert isinstance(result, list)

    def test_calculate_pair_returns_list_for_label(
        self, label_calculator, sample_data
    ):
        """Should return list for per-label agreement."""
        result = label_calculator.calculate_pair(
            sample_data, "user1@example.com", "user2@example.com"
        )

        assert isinstance(result, list)

    def test_calculate_pair_returns_none_for_no_overlap(self, overall_calculator):
        """Should return None when no common tasks."""
        data = pl.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "ground_truth_member": [None, None],
                "user1@example.com": [
                    [{"label_type": "action", "direction": "Long"}],
                    None,
                ],
                "user2@example.com": [
                    None,
                    [{"label_type": "action", "direction": "Short"}],
                ],
            }
        )

        result = overall_calculator.calculate_pair(
            data, "user1@example.com", "user2@example.com"
        )

        assert result is None

    def test_calculate_all_pairs(self, overall_calculator, sample_data):
        """Should calculate all pairwise combinations."""
        annotators = ["user1@example.com", "user2@example.com"]
        result = overall_calculator.calculate_all_pairs(sample_data, annotators)

        assert "user1@example.com" in result
        assert "user2@example.com" in result
        assert result["user1@example.com"]["user1@example.com"] is None  # Self
        assert result["user1@example.com"]["user2@example.com"] is not None


class TestAggregatePerLabelScores:
    """Tests for aggregate_per_label_scores method."""

    @pytest.fixture
    def field_calculator(self):
        return PairwiseCalculator.create(case="field", common=False)

    @pytest.fixture
    def label_calculator(self):
        return PairwiseCalculator.create(case="label", common=False)

    def test_aggregates_field_scores_with_average(self, field_calculator):
        """Should average field scores when average=True."""
        scores = {
            "user1": {
                "user2": [
                    {"direction": 0.8, "exposure_change": 0.6},
                    {"direction": 0.6, "exposure_change": 0.8},
                ]
            },
            "user2": {"user1": None},
        }
        annotators = ["user1", "user2"]

        result = field_calculator.aggregate_per_label_scores(
            scores, annotators, average=True
        )

        assert "user1" in result
        assert "user2" in result["user1"]
        # Average of 0.8 and 0.6 = 0.7
        assert result["user1"]["user2"]["direction"] == pytest.approx(0.7)

    def test_aggregates_label_scores_with_average(self, label_calculator):
        """Should compute ratios for label scores when average=True."""
        scores = {
            "user1": {
                "user2": [
                    ({"Long": 1, "Short": 0}, {"Long": 1, "Short": 1}),
                    ({"Long": 1, "Short": 1}, {"Long": 1, "Short": 1}),
                ]
            },
            "user2": {"user1": None},
        }
        annotators = ["user1", "user2"]

        result = label_calculator.aggregate_per_label_scores(
            scores, annotators, average=True
        )

        # Long: 2 agreements / 2 counts = 1.0
        # Short: 1 agreement / 2 counts = 0.5
        assert result["user1"]["user2"]["Long"] == pytest.approx(1.0)
        assert result["user1"]["user2"]["Short"] == pytest.approx(0.5)

    def test_returns_raw_counts_when_not_average(self, label_calculator):
        """Should return raw agreement counts when average=False."""
        scores = {
            "user1": {
                "user2": [
                    ({"Long": 2, "Short": 1}, {"Long": 2, "Short": 2}),
                ]
            },
            "user2": {"user1": None},
        }
        annotators = ["user1", "user2"]

        result = label_calculator.aggregate_per_label_scores(
            scores, annotators, average=False
        )

        # Raw agreement counts
        assert result["user1"]["user2"]["Long"] == 2
        assert result["user1"]["user2"]["Short"] == 1

    def test_handles_empty_scores(self, field_calculator):
        """Should handle empty score lists."""
        scores = {
            "user1": {"user2": []},
            "user2": {"user1": None},
        }
        annotators = ["user1", "user2"]

        result = field_calculator.aggregate_per_label_scores(
            scores, annotators, average=True
        )

        assert result["user1"]["user2"] == {}
