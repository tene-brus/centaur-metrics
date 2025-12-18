"""Tests for src/agreement/per_label.py."""

import pytest

from src.agreement.per_label import (
    ALL_LABEL_KEYS,
    PerLabelAgreementCalculator,
    get_all_label_keys,
    get_label_key,
)


class TestGetLabelKey:
    """Tests for get_label_key function."""

    def test_ambiguous_label_gets_field_context(self):
        """Ambiguous labels like 'Unclear' should get field context."""
        result = get_label_key("Unclear", "direction")
        assert result == "Unclear (direction)"

    def test_ambiguous_label_exposure_change(self):
        """Unclear in exposure_change should be disambiguated."""
        result = get_label_key("Unclear", "exposure_change")
        assert result == "Unclear (exposure_change)"

    def test_non_ambiguous_label_unchanged(self):
        """Non-ambiguous labels should be returned unchanged."""
        result = get_label_key("Long", "direction")
        assert result == "Long"

    def test_non_ambiguous_label_other_field(self):
        """Non-ambiguous labels in any field should be unchanged."""
        result = get_label_key("Increase", "exposure_change")
        assert result == "Increase"


class TestGetAllLabelKeys:
    """Tests for get_all_label_keys function."""

    def test_returns_list(self):
        """Should return a list."""
        result = get_all_label_keys()
        assert isinstance(result, list)

    def test_contains_disambiguated_unclear(self):
        """Should contain field-specific Unclear keys."""
        result = get_all_label_keys()

        assert "Unclear (direction)" in result
        assert "Unclear (exposure_change)" in result
        assert "Unclear (remaining_exposure)" in result

    def test_no_duplicate_keys(self):
        """Should not have duplicate keys."""
        result = get_all_label_keys()
        assert len(result) == len(set(result))

    def test_contains_standard_labels(self):
        """Should contain standard non-ambiguous labels."""
        result = get_all_label_keys()

        assert "Long" in result
        assert "Short" in result
        assert "Increase" in result
        assert "Decrease" in result


class TestAllLabelKeys:
    """Tests for ALL_LABEL_KEYS constant."""

    def test_all_label_keys_matches_function(self):
        """ALL_LABEL_KEYS should match get_all_label_keys() output."""
        assert ALL_LABEL_KEYS == get_all_label_keys()


class TestPerLabelAgreementCalculator:
    """Tests for PerLabelAgreementCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PerLabelAgreementCalculator()

    def test_returns_tuple(self, calculator, sample_trade_long):
        """Should return tuple of (agreements, counts)."""
        result = calculator.calculate([sample_trade_long], [sample_trade_long])

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_dicts_returned(self, calculator, sample_trade_long):
        """Should return two dictionaries."""
        agreements, counts = calculator.calculate(
            [sample_trade_long], [sample_trade_long]
        )

        assert isinstance(agreements, dict)
        assert isinstance(counts, dict)

    def test_both_empty_returns_empty_dicts(self, calculator):
        """Should return empty dicts when both lists empty."""
        agreements, counts = calculator.calculate([], [])

        assert agreements == {}
        assert counts == {}

    def test_first_empty_returns_zero_dicts(self, calculator, sample_trade_long):
        """Should return zero-filled dicts when first list empty."""
        agreements, counts = calculator.calculate([], [sample_trade_long])

        for key in ALL_LABEL_KEYS:
            assert key in agreements
            assert agreements[key] == 0

    def test_second_empty_returns_zero_dicts(self, calculator, sample_trade_long):
        """Should return zero-filled dicts when second list empty."""
        agreements, counts = calculator.calculate([sample_trade_long], [])

        for key in ALL_LABEL_KEYS:
            assert key in agreements
            assert agreements[key] == 0

    def test_identical_trades_count_agreements(self, calculator, sample_trade_long):
        """Identical trades should increment agreement counts."""
        agreements, counts = calculator.calculate(
            [sample_trade_long], [sample_trade_long]
        )

        # Long is the direction, should have agreement
        assert agreements["Long"] > 0

    def test_tracks_unclear_by_field(self, calculator, sample_trade_unclear):
        """Should track 'Unclear' separately per field."""
        agreements, counts = calculator.calculate(
            [sample_trade_unclear], [sample_trade_unclear]
        )

        # Unclear appears in direction, exposure_change, and remaining_exposure
        assert counts["Unclear (direction)"] > 0
        assert counts["Unclear (exposure_change)"] > 0
        assert counts["Unclear (remaining_exposure)"] > 0

    def test_different_labels_no_agreement(self, calculator):
        """Different labels should not count as agreement."""
        trade_a = {
            "asset_reference_type": "Majors",
            "direction": "Long",
            "label_type": "action",
        }
        trade_b = {
            "asset_reference_type": "Majors",
            "direction": "Short",  # Different
            "label_type": "action",
        }

        agreements, counts = calculator.calculate([trade_a], [trade_b])

        # Both Long and Short should have counts
        assert counts["Long"] > 0
        assert counts["Short"] > 0

        # Neither should have agreement (different labels)
        # Agreement only increments when both match


class TestPerLabelSimilarity:
    """Tests for per-label similarity method."""

    @pytest.fixture
    def calculator(self):
        return PerLabelAgreementCalculator()

    def test_returns_three_tuple(self, calculator, sample_trade_long):
        """Should return tuple of (agreement_dict, count_dict, overall_score)."""
        result = calculator.similarity(sample_trade_long, sample_trade_long)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_agreement_dict_has_all_keys(self, calculator, sample_trade_long):
        """Agreement dict should have all label keys."""
        agreement, count, score = calculator.similarity(
            sample_trade_long, sample_trade_long
        )

        for key in ALL_LABEL_KEYS:
            assert key in agreement

    def test_count_dict_has_all_keys(self, calculator, sample_trade_long):
        """Count dict should have all label keys."""
        agreement, count, score = calculator.similarity(
            sample_trade_long, sample_trade_long
        )

        for key in ALL_LABEL_KEYS:
            assert key in count

    def test_matching_labels_increment_agreement(self, calculator):
        """Matching labels should increment agreement count."""
        trade_a = {
            "direction": "Long",
            "exposure_change": "Increase",
            "label_type": "action",
        }
        trade_b = {
            "direction": "Long",
            "exposure_change": "Increase",
            "label_type": "action",
        }

        agreement, count, score = calculator.similarity(trade_a, trade_b)

        assert agreement["Long"] == 1
        assert agreement["Increase"] == 1
        assert agreement["action"] == 1

    def test_different_labels_no_agreement(self, calculator):
        """Different labels should not increment agreement."""
        trade_a = {"direction": "Long"}
        trade_b = {"direction": "Short"}

        agreement, count, score = calculator.similarity(trade_a, trade_b)

        assert agreement["Long"] == 0
        assert agreement["Short"] == 0

    def test_unclear_tracked_by_field(self, calculator):
        """Unclear should be tracked separately per field."""
        trade = {
            "direction": "Unclear",
            "exposure_change": "Unclear",
            "remaining_exposure": "Unclear",
        }

        agreement, count, score = calculator.similarity(trade, trade)

        # Each Unclear in different field should count separately
        assert count["Unclear (direction)"] == 1
        assert count["Unclear (exposure_change)"] == 1
        assert count["Unclear (remaining_exposure)"] == 1

        # And agreements should be tracked
        assert agreement["Unclear (direction)"] == 1
        assert agreement["Unclear (exposure_change)"] == 1
        assert agreement["Unclear (remaining_exposure)"] == 1
