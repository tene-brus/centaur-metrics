"""Tests for src/agreement/matching.py."""

import pytest

from src.agreement.matching import (
    _extract_score,
    find_best_matches,
    match_trades_by_group,
)


def simple_similarity(trade_a: dict, trade_b: dict) -> float:
    """Simple similarity function for testing."""
    if trade_a.get("direction") == trade_b.get("direction"):
        return 1.0
    return 0.0


def tuple_similarity(trade_a: dict, trade_b: dict) -> tuple:
    """Similarity function that returns tuple (for per_field/per_label)."""
    score = 1.0 if trade_a.get("direction") == trade_b.get("direction") else 0.0
    return ({"direction": score}, score)


class TestFindBestMatches:
    """Tests for find_best_matches function."""

    def test_matches_identical_trades(self):
        """Should match identical trades with score 1.0."""
        trades_a = [{"direction": "Long", "asset_reference_type": "Majors"}]
        trades_b = [{"direction": "Long", "asset_reference_type": "Majors"}]

        matches = find_best_matches(trades_a, trades_b, simple_similarity)

        assert len(matches) == 1
        assert matches[0][2] == 1.0

    def test_matches_best_pairs_greedily(self):
        """Should match pairs with highest similarity first."""
        trades_a = [
            {"direction": "Long", "id": "a1"},
            {"direction": "Short", "id": "a2"},
        ]
        trades_b = [
            {"direction": "Short", "id": "b1"},
            {"direction": "Long", "id": "b2"},
        ]

        matches = find_best_matches(trades_a, trades_b, simple_similarity)

        assert len(matches) == 2
        # Should match Long-Long and Short-Short
        directions_matched = [
            (m[0]["direction"], m[1]["direction"]) for m in matches
        ]
        assert ("Long", "Long") in directions_matched
        assert ("Short", "Short") in directions_matched

    def test_handles_unequal_counts(self):
        """Should handle different numbers of trades."""
        trades_a = [
            {"direction": "Long"},
            {"direction": "Short"},
            {"direction": "Long"},
        ]
        trades_b = [{"direction": "Long"}]

        matches = find_best_matches(trades_a, trades_b, simple_similarity)

        # Can only match 1 pair
        assert len(matches) == 1
        assert matches[0][2] == 1.0

    def test_returns_empty_for_empty_trades_a(self):
        """Should return empty list if trades_a is empty."""
        matches = find_best_matches([], [{"direction": "Long"}], simple_similarity)
        assert matches == []

    def test_returns_empty_for_empty_trades_b(self):
        """Should return empty list if trades_b is empty."""
        matches = find_best_matches([{"direction": "Long"}], [], simple_similarity)
        assert matches == []

    def test_handles_tuple_similarity(self):
        """Should work with similarity functions returning tuples."""
        trades_a = [{"direction": "Long"}]
        trades_b = [{"direction": "Long"}]

        matches = find_best_matches(trades_a, trades_b, tuple_similarity)

        assert len(matches) == 1
        assert isinstance(matches[0][2], tuple)
        assert matches[0][2][1] == 1.0


class TestMatchTradesByGroup:
    """Tests for match_trades_by_group function."""

    def test_matches_within_same_group(self):
        """Should match trades within the same primary key group."""
        trades_a = [
            {"direction": "Long", "asset_reference_type": "Majors"},
            {"direction": "Short", "asset_reference_type": "DeFi"},
        ]
        trades_b = [
            {"direction": "Long", "asset_reference_type": "Majors"},
            {"direction": "Long", "asset_reference_type": "DeFi"},
        ]

        matches = match_trades_by_group(trades_a, trades_b, simple_similarity)

        # Should have 2 matches (one per group)
        assert len(matches) == 2

    def test_groups_by_specific_assets(self):
        """Should group by specific assets for Specific Asset(s) type."""
        trades_a = [
            {
                "direction": "Long",
                "asset_reference_type": "Specific Asset(s)",
                "specific_assets": ["BTC"],
            },
            {
                "direction": "Short",
                "asset_reference_type": "Specific Asset(s)",
                "specific_assets": ["ETH"],
            },
        ]
        trades_b = [
            {
                "direction": "Short",
                "asset_reference_type": "Specific Asset(s)",
                "specific_assets": ["BTC"],
            },
            {
                "direction": "Short",
                "asset_reference_type": "Specific Asset(s)",
                "specific_assets": ["ETH"],
            },
        ]

        matches = match_trades_by_group(trades_a, trades_b, simple_similarity)

        # BTC group: Long vs Short = 0.0
        # ETH group: Short vs Short = 1.0
        assert len(matches) == 2

    def test_handles_unmatched_groups(self):
        """Should handle groups that only appear in one trade list."""
        trades_a = [{"direction": "Long", "asset_reference_type": "Majors"}]
        trades_b = [{"direction": "Long", "asset_reference_type": "DeFi"}]

        matches = match_trades_by_group(trades_a, trades_b, simple_similarity)

        # No overlapping groups
        assert len(matches) == 0

    def test_handles_empty_lists(self):
        """Should handle empty trade lists."""
        matches = match_trades_by_group([], [], simple_similarity)
        assert matches == []


class TestExtractScore:
    """Tests for _extract_score helper function."""

    def test_extracts_float(self):
        """Should return float as-is."""
        assert _extract_score(0.75) == 0.75

    def test_extracts_from_tuple_length_2(self):
        """Should extract score from 2-tuple (per_field format)."""
        similarity = ({"field": 0.5}, 0.8)
        assert _extract_score(similarity) == 0.8

    def test_extracts_from_tuple_length_3(self):
        """Should extract score from 3-tuple (per_label format)."""
        similarity = ({"label": 1}, {"label": 2}, 0.9)
        assert _extract_score(similarity) == 0.9

    def test_handles_zero_score(self):
        """Should handle zero scores."""
        assert _extract_score(0.0) == 0.0
        assert _extract_score(({}, {}, 0.0)) == 0.0
