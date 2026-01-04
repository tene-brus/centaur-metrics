"""Tests for src/metrics/unified_pairwise.py."""

import polars as pl
import pytest

from src.metrics.unified_pairwise import (
    AggregatedScores,
    AllPairScores,
    UnifiedPairwiseCalculator,
    validate_and_dump_annotations,
)
from src.models.constants import AGREEMENT_FIELDS, ALL_LABEL_KEYS


class TestValidateAndDumpAnnotations:
    """Tests for validate_and_dump_annotations function."""

    def test_returns_empty_list_for_none(self):
        """Returns empty list when input is None."""
        result = validate_and_dump_annotations(None)
        assert result == []

    def test_returns_empty_list_for_empty_list(self):
        """Returns empty list when input is empty list."""
        result = validate_and_dump_annotations([])
        assert result == []

    def test_validates_valid_annotations(self):
        """Returns validated annotations as dicts."""
        annotations = [
            {
                "label_type": "action",
                "asset_reference_type": "Majors",
                "direction": "Long",
                "action_exposure_change": "Increase",
                "action_position_status": "Clearly a new position",
            }
        ]
        result = validate_and_dump_annotations(annotations)

        assert len(result) == 1
        assert result[0]["label_type"] == "action"
        assert result[0]["direction"] == "Long"

    def test_skips_invalid_annotations(self):
        """Skips annotations that fail validation."""
        annotations = [
            {"invalid_field": "invalid_value"},  # Invalid
            {
                "label_type": "action",
                "asset_reference_type": "Majors",
                "direction": "Long",
                "action_exposure_change": "Increase",
                "action_position_status": "Clearly a new position",
            },  # Valid
        ]
        result = validate_and_dump_annotations(annotations)

        # Only the valid annotation should be returned
        assert len(result) == 1


class TestAggregatedScores:
    """Tests for AggregatedScores dataclass."""

    def test_dataclass_fields(self):
        """AggregatedScores has expected fields."""
        scores = AggregatedScores(
            overall=0.8,
            per_field={"direction": 0.9},
            per_label_ratios={"Long": 0.95},
            per_label_counts={"Long": 10.0},
            num_tasks=5,
        )

        assert scores.overall == 0.8
        assert scores.per_field == {"direction": 0.9}
        assert scores.per_label_ratios == {"Long": 0.95}
        assert scores.per_label_counts == {"Long": 10.0}
        assert scores.num_tasks == 5


class TestAllPairScores:
    """Tests for AllPairScores dataclass."""

    def test_dataclass_fields(self):
        """AllPairScores has expected fields."""
        scores = AllPairScores(
            scores={"annotator1": {"annotator2": None}},
            annotators=["annotator1", "annotator2"],
        )

        assert scores.scores == {"annotator1": {"annotator2": None}}
        assert scores.annotators == ["annotator1", "annotator2"]


class TestUnifiedPairwiseCalculator:
    """Tests for UnifiedPairwiseCalculator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with two annotators."""
        return pl.DataFrame(
            {
                "task_id": ["task1", "task2"],
                "trader": ["trader1", "trader1"],
                "annotator1": [
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
                            "label_type": "state",
                            "asset_reference_type": "DeFi",
                            "direction": "Short",
                            "state_type": "Explicit State",
                            "remaining_exposure": "Some",
                            "state_exposure_change": "No Change",
                            "state_position_status": "Clearly an existing position",
                        }
                    ],
                ],
                "annotator2": [
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
                            "label_type": "state",
                            "asset_reference_type": "DeFi",
                            "direction": "Short",
                            "state_type": "Explicit State",
                            "remaining_exposure": "Some",
                            "state_exposure_change": "No Change",
                            "state_position_status": "Clearly an existing position",
                        }
                    ],
                ],
            }
        )

    @pytest.fixture
    def different_annotations_data(self):
        """Create data where annotators disagree."""
        return pl.DataFrame(
            {
                "task_id": ["task1"],
                "trader": ["trader1"],
                "annotator1": [
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ]
                ],
                "annotator2": [
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Short",  # Different
                            "action_exposure_change": "Decrease",  # Different
                            "action_position_status": "Clearly an existing position",  # Different
                        }
                    ]
                ],
            }
        )

    def test_init_with_common_false(self):
        """Initialize with common=False by default."""
        calculator = UnifiedPairwiseCalculator()
        assert calculator.common is False

    def test_init_with_common_true(self):
        """Initialize with common=True."""
        calculator = UnifiedPairwiseCalculator(common=True)
        assert calculator.common is True

    def test_calculate_all_pairs_returns_all_pair_scores(self, sample_data):
        """calculate_all_pairs returns AllPairScores."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        assert isinstance(result, AllPairScores)
        assert result.annotators == annotators

    def test_self_comparison_is_none(self, sample_data):
        """Self-comparison scores are None."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        assert result.scores["annotator1"]["annotator1"] is None
        assert result.scores["annotator2"]["annotator2"] is None

    def test_identical_annotations_have_high_agreement(self, sample_data):
        """Identical annotations should have high agreement score."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert scores.overall == 1.0

    def test_different_annotations_have_lower_agreement(
        self, different_annotations_data
    ):
        """Different annotations should have lower agreement score."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(different_annotations_data, annotators)

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert scores.overall < 1.0

    def test_aggregated_scores_has_per_field(self, sample_data):
        """AggregatedScores contains per_field dict."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert isinstance(scores.per_field, dict)
        for field in AGREEMENT_FIELDS:
            assert field in scores.per_field

    def test_aggregated_scores_has_per_label_ratios(self, sample_data):
        """AggregatedScores contains per_label_ratios dict."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert isinstance(scores.per_label_ratios, dict)

    def test_aggregated_scores_has_per_label_counts(self, sample_data):
        """AggregatedScores contains per_label_counts dict."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert isinstance(scores.per_label_counts, dict)

    def test_num_tasks_is_correct(self, sample_data):
        """num_tasks reflects number of compared tasks."""
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(sample_data, annotators)

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert scores.num_tasks == 2

    def test_returns_none_for_no_common_tasks(self):
        """Returns None when annotators have no common tasks."""
        data = pl.DataFrame(
            {
                "task_id": ["task1", "task2"],
                "trader": ["trader1", "trader1"],
                "annotator1": [
                    [{"label_type": "action", "direction": "Long"}],
                    None,
                ],
                "annotator2": [
                    None,
                    [{"label_type": "action", "direction": "Short"}],
                ],
            }
        )
        calculator = UnifiedPairwiseCalculator()
        annotators = ["annotator1", "annotator2"]

        result = calculator.calculate_all_pairs(data, annotators)

        assert result.scores["annotator1"]["annotator2"] is None

    def test_ground_truth_special_case(self):
        """Ground truth comparisons exclude ground truth member."""
        data = pl.DataFrame(
            {
                "task_id": ["task1"],
                "trader": ["trader1"],
                "ground_truth_member": ["annotator1"],
                "annotator1": [
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                        }
                    ]
                ],
                "ground_truth": [
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                        }
                    ]
                ],
            }
        )
        calculator = UnifiedPairwiseCalculator(common=False)
        annotators = ["annotator1", "ground_truth"]

        result = calculator.calculate_all_pairs(data, annotators)

        # annotator1 was the GT member, so should be excluded from comparison
        assert result.scores["annotator1"]["ground_truth"] is None


class TestAggregateTaskResults:
    """Tests for _aggregate_task_results method."""

    def test_averages_overall_scores(self):
        """Overall score is averaged across tasks."""
        # Create calculator and mock task results
        calculator = UnifiedPairwiseCalculator()

        # We need to call _calculate_pair with data that produces multiple tasks
        data = pl.DataFrame(
            {
                "task_id": ["task1", "task2", "task3"],
                "trader": ["trader1", "trader1", "trader1"],
                "annotator1": [
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
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ],
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ],
                ],
                "annotator2": [
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
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ],
                    [
                        {
                            "label_type": "action",
                            "asset_reference_type": "Majors",
                            "direction": "Long",
                            "action_exposure_change": "Increase",
                            "action_position_status": "Clearly a new position",
                        }
                    ],
                ],
            }
        )

        result = calculator.calculate_all_pairs(data, ["annotator1", "annotator2"])

        scores = result.scores["annotator1"]["annotator2"]
        assert scores is not None
        assert scores.num_tasks == 3
        # All identical, so overall should be 1.0
        assert scores.overall == 1.0
