"""Tests for src/agreement/overall.py."""

import pytest

from src.agreement.overall import OverallAgreementCalculator
from src.models.constants import PRIMARY_KEY_WEIGHT, REMAINING_FIELDS_WEIGHT


class TestOverallAgreementCalculator:
    """Tests for OverallAgreementCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return OverallAgreementCalculator()

    def test_both_empty_returns_one(self, calculator):
        """Should return 1.0 when both trade lists are empty."""
        result = calculator.calculate([], [])
        assert result == 1.0

    def test_first_empty_returns_zero(self, calculator, sample_trade_long):
        """Should return 0.0 when first trade list is empty."""
        result = calculator.calculate([], [sample_trade_long])
        assert result == 0.0

    def test_second_empty_returns_zero(self, calculator, sample_trade_long):
        """Should return 0.0 when second trade list is empty."""
        result = calculator.calculate([sample_trade_long], [])
        assert result == 0.0

    def test_identical_trades_high_score(self, calculator, sample_trade_long):
        """Should return high score for identical trades."""
        result = calculator.calculate([sample_trade_long], [sample_trade_long])

        # Should be PRIMARY_KEY_WEIGHT + REMAINING_FIELDS_WEIGHT (= 1.0)
        assert result == pytest.approx(1.0)

    def test_different_trades_lower_score(
        self, calculator, sample_trade_long, sample_trade_short
    ):
        """Should return lower score for different trades with same asset ref."""
        result = calculator.calculate([sample_trade_long], [sample_trade_short])

        # Same asset reference, but different fields
        # Score should be less than 1.0
        assert result < 1.0
        assert result > 0.0

    def test_no_matching_assets_low_score(
        self, calculator, sample_trade_long, sample_trade_state
    ):
        """Should return lower score when asset references don't match."""
        # sample_trade_long has Specific Asset(s), sample_trade_state has Majors
        result = calculator.calculate([sample_trade_long], [sample_trade_state])

        # No matching primary key groups, so no matches found
        assert result == 0.0

    def test_normalized_by_max_trades(self, calculator, sample_trade_long):
        """Score should be normalized by max number of trades."""
        trades_a = [sample_trade_long, sample_trade_long]
        trades_b = [sample_trade_long]

        result = calculator.calculate(trades_a, trades_b)

        # Only 1 match possible, normalized by max(2, 1) = 2
        assert result == pytest.approx(0.5)

    def test_multiple_trades_multiple_matches(self, calculator):
        """Should handle multiple trades with multiple matches."""
        trade_1 = {
            "asset_reference_type": "Majors",
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }
        trade_2 = {
            "asset_reference_type": "DeFi",
            "direction": "Short",
            "state_type": "Direct State",
            "exposure_change": "Decrease",
            "position_status": "Clearly an existing position",
            "optional_task_flags": [],
        }

        result = calculator.calculate([trade_1, trade_2], [trade_1, trade_2])

        # Both trades should match perfectly
        assert result == pytest.approx(1.0)


class TestSimilarityMethod:
    """Tests for the similarity method."""

    @pytest.fixture
    def calculator(self):
        return OverallAgreementCalculator()

    def test_identical_trades_score_one(self, calculator, sample_trade_long):
        """Identical trades should have similarity score of 1.0."""
        score = calculator.similarity(sample_trade_long, sample_trade_long)
        assert score == pytest.approx(1.0)

    def test_different_direction_reduces_score(self, calculator):
        """Different direction should reduce similarity score."""
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

        score = calculator.similarity(trade_a, trade_b)
        assert score == pytest.approx(0.8)  # 4/5 fields match

    def test_all_different_fields_score_zero(self, calculator):
        """Trades with all different fields should have low score."""
        trade_a = {
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1"],
        }
        trade_b = {
            "direction": "Short",
            "state_type": "Direct State",
            "exposure_change": "Decrease",
            "position_status": "Clearly an existing position",
            "optional_task_flags": ["flag2"],
        }

        score = calculator.similarity(trade_a, trade_b)
        assert score == pytest.approx(0.0)

    def test_optional_flags_partial_match(self, calculator):
        """Partial match on optional flags should give partial score."""
        trade_a = {
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1", "flag2"],
        }
        trade_b = {
            "direction": "Long",
            "state_type": "Explicit State",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1", "flag3"],  # 1 of 2 match
        }

        score = calculator.similarity(trade_a, trade_b)
        # 4 fields match perfectly, optional_task_flags has 1/2 intersection
        # Score = (4 + 0.5) / 5 = 0.9
        assert score == pytest.approx(0.9)

    def test_both_empty_flags_match(self, calculator):
        """Both trades with empty optional_task_flags should match."""
        trade_a = {
            "direction": "Long",
            "state_type": None,
            "exposure_change": None,
            "position_status": None,
            "optional_task_flags": [],
        }
        trade_b = {
            "direction": "Long",
            "state_type": None,
            "exposure_change": None,
            "position_status": None,
            "optional_task_flags": [],
        }

        score = calculator.similarity(trade_a, trade_b)
        # All None fields match, empty flags match
        assert score == pytest.approx(1.0)
