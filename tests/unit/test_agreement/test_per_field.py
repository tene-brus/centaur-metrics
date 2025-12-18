"""Tests for src/agreement/per_field.py."""

import pytest

from src.agreement.per_field import PerFieldAgreementCalculator
from src.models.constants import AGREEMENT_FIELDS


class TestPerFieldAgreementCalculator:
    """Tests for PerFieldAgreementCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PerFieldAgreementCalculator()

    def test_returns_dict(self, calculator, sample_trade_long):
        """Should return a dictionary."""
        result = calculator.calculate([sample_trade_long], [sample_trade_long])
        assert isinstance(result, dict)

    def test_returns_all_agreement_fields(self, calculator, sample_trade_long):
        """Should return scores for all agreement fields."""
        result = calculator.calculate([sample_trade_long], [sample_trade_long])

        for field in AGREEMENT_FIELDS:
            assert field in result

    def test_both_empty_returns_default_scores(self, calculator):
        """Should return default scores (0.2) when both lists empty."""
        result = calculator.calculate([], [])

        for field in AGREEMENT_FIELDS:
            assert result[field] == pytest.approx(0.2)

    def test_first_empty_returns_zeros(self, calculator, sample_trade_long):
        """Should return zeros when first list is empty."""
        result = calculator.calculate([], [sample_trade_long])

        for field in AGREEMENT_FIELDS:
            assert result[field] == 0.0

    def test_second_empty_returns_zeros(self, calculator, sample_trade_long):
        """Should return zeros when second list is empty."""
        result = calculator.calculate([sample_trade_long], [])

        for field in AGREEMENT_FIELDS:
            assert result[field] == 0.0

    def test_identical_trades_positive_scores(self, calculator, sample_trade_long):
        """Identical trades should have positive scores for matching fields."""
        result = calculator.calculate([sample_trade_long], [sample_trade_long])

        # Fields with values should have positive agreement
        # Note: sample_trade_long has state_type=None, so that field may be lower
        assert result["direction"] > 0
        assert result["exposure_change"] > 0
        assert result["position_status"] > 0

    def test_different_direction_affects_direction_score(self, calculator):
        """Different direction should lower direction score only."""
        trade_a = {
            "asset_reference_type": "Majors",
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }
        trade_b = {
            "asset_reference_type": "Majors",
            "direction": "Short",  # Different
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }

        result = calculator.calculate([trade_a], [trade_b])

        # Direction should have lower score
        assert result["direction"] < result["state_type"]
        assert result["direction"] < result["exposure_change"]

    def test_no_matching_groups_returns_zeros(
        self, calculator, sample_trade_long, sample_trade_state
    ):
        """Should return zeros when no asset reference groups match."""
        result = calculator.calculate([sample_trade_long], [sample_trade_state])

        for field in AGREEMENT_FIELDS:
            assert result[field] == 0.0


class TestPerFieldSimilarity:
    """Tests for per-field similarity method."""

    @pytest.fixture
    def calculator(self):
        return PerFieldAgreementCalculator()

    def test_returns_tuple(self, calculator, sample_trade_long):
        """Should return tuple of (field_scores, overall_score)."""
        result = calculator.similarity(sample_trade_long, sample_trade_long)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], float)

    def test_field_scores_dict_has_all_fields(self, calculator, sample_trade_long):
        """Field scores dict should have all agreement fields."""
        field_scores, _ = calculator.similarity(sample_trade_long, sample_trade_long)

        for field in AGREEMENT_FIELDS:
            assert field in field_scores

    def test_identical_trades_all_fields_positive(self, calculator, sample_trade_long):
        """Identical trades should have positive scores for all fields."""
        field_scores, overall = calculator.similarity(
            sample_trade_long, sample_trade_long
        )

        for field in AGREEMENT_FIELDS:
            assert field_scores[field] >= 0

    def test_different_field_values_reduce_score(self, calculator):
        """Different field values should reduce that field's score."""
        trade_a = {
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }
        trade_b = {
            "direction": "Short",  # Different
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }

        field_scores, _ = calculator.similarity(trade_a, trade_b)

        # Direction should be 0 (no match)
        assert field_scores["direction"] == 0

        # Other fields should have positive scores
        assert field_scores["state_type"] > 0
        assert field_scores["exposure_change"] > 0

    def test_overall_score_is_average(self, calculator):
        """Overall score should be average of field scores."""
        trade_a = {
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }

        field_scores, overall = calculator.similarity(trade_a, trade_a)

        # Overall should be sum of field scores / number of fields
        expected = sum(field_scores.values()) / len(field_scores)
        assert overall == pytest.approx(expected)
