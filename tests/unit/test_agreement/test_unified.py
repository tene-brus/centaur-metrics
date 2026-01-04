"""Tests for src/agreement/unified.py."""

import pytest

from src.agreement.unified import (
    UnifiedAgreementResult,
    UnifiedSimilarity,
    calculate_unified_agreement,
    unified_similarity,
)
from src.models.constants import AGREEMENT_FIELDS


class TestUnifiedSimilarity:
    """Tests for unified_similarity function."""

    @pytest.fixture
    def identical_trades(self):
        """Create identical trade pair."""
        trade = {
            "state_type": "Explicit State",
            "direction": "Long",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1", "flag2"],
            "label_type": "action",
            "asset_reference_type": "Specific Asset(s)",
            "remaining_exposure": "Some",
        }
        return trade.copy(), trade.copy()

    @pytest.fixture
    def different_trades(self):
        """Create completely different trade pair."""
        trade_a = {
            "state_type": "Explicit State",
            "direction": "Long",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1"],
        }
        trade_b = {
            "state_type": "Direct State",
            "direction": "Short",
            "exposure_change": "Decrease",
            "position_status": "Clearly an existing position",
            "optional_task_flags": ["flag2"],
        }
        return trade_a, trade_b

    def test_returns_unified_similarity_dataclass(self, identical_trades):
        """unified_similarity returns UnifiedSimilarity dataclass."""
        trade_a, trade_b = identical_trades
        result = unified_similarity(trade_a, trade_b)

        assert isinstance(result, UnifiedSimilarity)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "field_scores")
        assert hasattr(result, "label_agreements")
        assert hasattr(result, "label_counts")

    def test_identical_trades_have_perfect_score(self, identical_trades):
        """Identical trades should have overall_score of 1.0."""
        trade_a, trade_b = identical_trades
        result = unified_similarity(trade_a, trade_b)

        assert result.overall_score == 1.0

    def test_different_trades_have_zero_score(self, different_trades):
        """Completely different trades should have overall_score of 0.0."""
        trade_a, trade_b = different_trades
        result = unified_similarity(trade_a, trade_b)

        assert result.overall_score == 0.0

    def test_field_scores_contain_all_agreement_fields(self, identical_trades):
        """field_scores should contain all AGREEMENT_FIELDS."""
        trade_a, trade_b = identical_trades
        result = unified_similarity(trade_a, trade_b)

        for field in AGREEMENT_FIELDS:
            assert field in result.field_scores

    def test_field_scores_sum_to_overall_for_identical(self, identical_trades):
        """For identical trades, field_scores should sum to 1.0."""
        trade_a, trade_b = identical_trades
        result = unified_similarity(trade_a, trade_b)

        # Each field contributes 1/5 = 0.2 when matched
        total = sum(result.field_scores.values())
        assert abs(total - 1.0) < 0.001

    def test_label_agreements_tracks_matches(self, identical_trades):
        """label_agreements should track matching labels."""
        trade_a, trade_b = identical_trades
        result = unified_similarity(trade_a, trade_b)

        # Explicit State should be counted as agreement
        assert result.label_agreements.get("Explicit State", 0) == 1
        assert result.label_agreements.get("Long", 0) == 1
        assert result.label_agreements.get("Increase", 0) == 1

    def test_label_counts_tracks_occurrences(self, identical_trades):
        """label_counts should track label occurrences."""
        trade_a, trade_b = identical_trades
        result = unified_similarity(trade_a, trade_b)

        # Each label appears once (both agree, so only counted once)
        assert result.label_counts.get("Explicit State", 0) == 1
        assert result.label_counts.get("Long", 0) == 1

    def test_partial_match_has_intermediate_score(self):
        """Partial match should have intermediate overall_score."""
        trade_a = {
            "state_type": "Explicit State",
            "direction": "Long",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": [],
        }
        trade_b = {
            "state_type": "Explicit State",  # Match
            "direction": "Long",  # Match
            "exposure_change": "Decrease",  # No match
            "position_status": "Clearly an existing position",  # No match
            "optional_task_flags": [],  # Match (both empty)
        }
        result = unified_similarity(trade_a, trade_b)

        # 3 out of 5 fields match
        assert 0.5 < result.overall_score < 0.7

    def test_optional_flags_partial_match(self):
        """Optional flags with partial overlap should have intermediate score."""
        trade_a = {
            "state_type": "Explicit State",
            "direction": "Long",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1", "flag2", "flag3"],
        }
        trade_b = {
            "state_type": "Explicit State",
            "direction": "Long",
            "exposure_change": "Increase",
            "position_status": "Clearly a new position",
            "optional_task_flags": ["flag1", "flag4"],  # 1 out of 3 overlap
        }
        result = unified_similarity(trade_a, trade_b)

        # 4 fields match perfectly, flags have 1/3 overlap
        assert 0.8 < result.overall_score < 1.0


class TestCalculateUnifiedAgreement:
    """Tests for calculate_unified_agreement function."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade lists."""
        trades_a = [
            {
                "state_type": "Explicit State",
                "direction": "Long",
                "exposure_change": "Increase",
                "position_status": "Clearly a new position",
                "optional_task_flags": [],
                "asset_reference_type": "Majors",
            }
        ]
        trades_b = [
            {
                "state_type": "Explicit State",
                "direction": "Long",
                "exposure_change": "Increase",
                "position_status": "Clearly a new position",
                "optional_task_flags": [],
                "asset_reference_type": "Majors",
            }
        ]
        return trades_a, trades_b

    def test_returns_unified_agreement_result(self, sample_trades):
        """calculate_unified_agreement returns UnifiedAgreementResult."""
        trades_a, trades_b = sample_trades
        result = calculate_unified_agreement(trades_a, trades_b)

        assert isinstance(result, UnifiedAgreementResult)
        assert hasattr(result, "overall")
        assert hasattr(result, "per_field")
        assert hasattr(result, "label_agreements")
        assert hasattr(result, "label_counts")
        assert hasattr(result, "num_matches")

    def test_both_empty_returns_perfect_score(self):
        """Both empty trade lists should return overall=1.0."""
        result = calculate_unified_agreement([], [])

        assert result.overall == 1.0
        assert result.num_matches == 0

    def test_only_a_empty_returns_zero(self):
        """Only trades_a empty should return overall=0.0."""
        trades_b = [{"state_type": "Explicit State", "direction": "Long"}]
        result = calculate_unified_agreement([], trades_b)

        assert result.overall == 0.0

    def test_only_b_empty_returns_zero(self):
        """Only trades_b empty should return overall=0.0."""
        trades_a = [{"state_type": "Explicit State", "direction": "Long"}]
        result = calculate_unified_agreement(trades_a, [])

        assert result.overall == 0.0

    def test_identical_trades_have_high_score(self, sample_trades):
        """Identical trade lists should have high overall score."""
        trades_a, trades_b = sample_trades
        result = calculate_unified_agreement(trades_a, trades_b)

        # Should be 1.0 (perfect match)
        assert result.overall == 1.0

    def test_num_matches_equals_matched_pairs(self, sample_trades):
        """num_matches should equal number of matched trade pairs."""
        trades_a, trades_b = sample_trades
        result = calculate_unified_agreement(trades_a, trades_b)

        assert result.num_matches == 1

    def test_per_field_contains_all_fields(self, sample_trades):
        """per_field should contain all AGREEMENT_FIELDS."""
        trades_a, trades_b = sample_trades
        result = calculate_unified_agreement(trades_a, trades_b)

        for field in AGREEMENT_FIELDS:
            assert field in result.per_field

    def test_multiple_trades_matched_correctly(self):
        """Multiple trades should be matched by primary key."""
        trades_a = [
            {
                "direction": "Long",
                "asset_reference_type": "Majors",
                "state_type": "Explicit State",
            },
            {
                "direction": "Short",
                "asset_reference_type": "DeFi",
                "state_type": "Direct State",
            },
        ]
        trades_b = [
            {
                "direction": "Long",
                "asset_reference_type": "Majors",
                "state_type": "Explicit State",
            },
            {
                "direction": "Short",
                "asset_reference_type": "DeFi",
                "state_type": "Direct State",
            },
        ]
        result = calculate_unified_agreement(trades_a, trades_b)

        assert result.num_matches == 2

    def test_unmatched_trades_lower_score(self):
        """More trades in one list should lower overall score."""
        trades_a = [
            {"direction": "Long", "asset_reference_type": "Majors"},
            {"direction": "Short", "asset_reference_type": "DeFi"},
        ]
        trades_b = [
            {"direction": "Long", "asset_reference_type": "Majors"},
        ]
        result = calculate_unified_agreement(trades_a, trades_b)

        # Only 1 match out of 2 trades in trades_a
        assert result.overall < 1.0
        assert result.num_matches == 1

    def test_tracks_trade_counts(self, sample_trades):
        """Should track trades_a_count and trades_b_count."""
        trades_a, trades_b = sample_trades
        result = calculate_unified_agreement(trades_a, trades_b)

        assert result.trades_a_count == 1
        assert result.trades_b_count == 1
