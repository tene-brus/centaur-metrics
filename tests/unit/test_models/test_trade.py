"""Tests for src/models/trade.py."""

import copy

import pytest

from src.models.trade import (
    get_primary_key,
    group_trades_by_key,
    normalize_annotations,
    normalize_exposure_change,
    normalize_optional_task_flags,
    normalize_position_status,
)


class TestNormalizePositionStatus:
    """Tests for normalize_position_status function."""

    def test_normalizes_action_position_status(self):
        """Should convert action_position_status to position_status."""
        annotations = [{"action_position_status": "Clearly a new position"}]
        result = normalize_position_status(annotations)

        assert "position_status" in result[0]
        assert result[0]["position_status"] == "Clearly a new position"
        assert "action_position_status" not in result[0]

    def test_normalizes_state_position_status(self):
        """Should convert state_position_status to position_status."""
        annotations = [{"state_position_status": "Clearly an existing position"}]
        result = normalize_position_status(annotations)

        assert "position_status" in result[0]
        assert result[0]["position_status"] == "Clearly an existing position"
        assert "state_position_status" not in result[0]

    def test_handles_empty_list(self):
        """Should handle empty annotation list."""
        result = normalize_position_status([])
        assert result == []

    def test_preserves_other_fields(self):
        """Should preserve fields other than position_status variants."""
        annotations = [
            {
                "action_position_status": "Clearly a new position",
                "direction": "Long",
                "label_type": "action",
            }
        ]
        result = normalize_position_status(annotations)

        assert result[0]["direction"] == "Long"
        assert result[0]["label_type"] == "action"


class TestNormalizeExposureChange:
    """Tests for normalize_exposure_change function."""

    def test_normalizes_action_exposure_change(self):
        """Should convert action_exposure_change to exposure_change."""
        annotations = [{"action_exposure_change": "Increase"}]
        result = normalize_exposure_change(annotations)

        assert "exposure_change" in result[0]
        assert result[0]["exposure_change"] == "Increase"
        assert "action_exposure_change" not in result[0]

    def test_normalizes_state_exposure_change(self):
        """Should convert state_exposure_change to exposure_change."""
        annotations = [{"state_exposure_change": "No Change"}]
        result = normalize_exposure_change(annotations)

        assert "exposure_change" in result[0]
        assert result[0]["exposure_change"] == "No Change"
        assert "state_exposure_change" not in result[0]


class TestNormalizeOptionalTaskFlags:
    """Tests for normalize_optional_task_flags function."""

    def test_normalizes_action_optional_flags(self):
        """Should convert action_optional_task_flags to optional_task_flags."""
        annotations = [{"action_optional_task_flags": ["flag1", "flag2"]}]
        result = normalize_optional_task_flags(annotations)

        assert "optional_task_flags" in result[0]
        assert result[0]["optional_task_flags"] == ["flag1", "flag2"]
        assert "action_optional_task_flags" not in result[0]

    def test_normalizes_state_optional_flags_with_retro(self):
        """Should merge state_total_retro_flag into optional_task_flags."""
        annotations = [
            {
                "state_optional_task_flags": ["flag1"],
                "state_total_retro_flag": "retro_flag",
            }
        ]
        result = normalize_optional_task_flags(annotations)

        assert "optional_task_flags" in result[0]
        assert "flag1" in result[0]["optional_task_flags"]
        assert "retro_flag" in result[0]["optional_task_flags"]
        assert "state_optional_task_flags" not in result[0]
        assert "state_total_retro_flag" not in result[0]

    def test_handles_none_flags(self):
        """Should convert None flags to empty list."""
        annotations = [{"action_optional_task_flags": None}]
        result = normalize_optional_task_flags(annotations)

        assert result[0]["optional_task_flags"] == []


class TestNormalizeAnnotations:
    """Tests for normalize_annotations function."""

    def test_normalizes_all_fields(self, sample_raw_annotation_action):
        """Should normalize all prefixed fields in one call."""
        annotations = [copy.deepcopy(sample_raw_annotation_action)]
        result = normalize_annotations(annotations)

        assert "position_status" in result[0]
        assert "exposure_change" in result[0]
        assert "optional_task_flags" in result[0]
        assert "action_position_status" not in result[0]
        assert "action_exposure_change" not in result[0]
        assert "action_optional_task_flags" not in result[0]

    def test_normalizes_state_annotation(self, sample_raw_annotation_state):
        """Should normalize state-type annotation including retro flag."""
        annotations = [copy.deepcopy(sample_raw_annotation_state)]
        result = normalize_annotations(annotations)

        assert "position_status" in result[0]
        assert result[0]["position_status"] == "Clearly an existing position"
        assert "exposure_change" in result[0]
        assert result[0]["exposure_change"] == "No Change"
        assert "retro_flag" in result[0]["optional_task_flags"]

    def test_handles_multiple_annotations(self):
        """Should normalize multiple annotations in a list."""
        annotations = [
            {"action_position_status": "Clearly a new position"},
            {"state_position_status": "Clearly an existing position"},
        ]
        result = normalize_annotations(annotations)

        assert len(result) == 2
        assert result[0]["position_status"] == "Clearly a new position"
        assert result[1]["position_status"] == "Clearly an existing position"


class TestGetPrimaryKey:
    """Tests for get_primary_key function."""

    def test_specific_assets_key(self, sample_trade_long):
        """Should generate key with sorted specific assets."""
        key = get_primary_key(sample_trade_long)

        assert key[0] == "Specific Asset(s)"
        assert key[1] == ("BTC", "ETH")  # Sorted

    def test_specific_assets_key_sorts_assets(self):
        """Should sort specific assets for consistent keys."""
        trade = {
            "asset_reference_type": "Specific Asset(s)",
            "specific_assets": ["ETH", "BTC", "SOL"],
        }
        key = get_primary_key(trade)

        assert key[1] == ("BTC", "ETH", "SOL")

    def test_non_specific_assets_key(self, sample_trade_state):
        """Should generate key without specific assets for non-specific types."""
        key = get_primary_key(sample_trade_state)

        assert key[0] == "Majors"
        assert key[1] is None

    def test_empty_specific_assets(self):
        """Should handle empty specific assets list."""
        trade = {"asset_reference_type": "Specific Asset(s)", "specific_assets": []}
        key = get_primary_key(trade)

        assert key[0] == "Specific Asset(s)"
        assert key[1] == ()

    def test_none_specific_assets(self):
        """Should handle None specific assets."""
        trade = {"asset_reference_type": "Specific Asset(s)", "specific_assets": None}
        key = get_primary_key(trade)

        assert key[0] == "Specific Asset(s)"
        assert key[1] == ()


class TestGroupTradesByKey:
    """Tests for group_trades_by_key function."""

    def test_groups_by_asset_reference(self, sample_trade_long, sample_trade_state):
        """Should group trades by their primary keys."""
        trades = [sample_trade_long, sample_trade_state]
        grouped = group_trades_by_key(trades)

        assert len(grouped) == 2
        assert ("Specific Asset(s)", ("BTC", "ETH")) in grouped
        assert ("Majors", None) in grouped

    def test_multiple_trades_same_key(self, sample_trade_long, sample_trade_short):
        """Should group multiple trades with same key together."""
        # Both have same asset reference
        trades = [sample_trade_long, sample_trade_short]
        grouped = group_trades_by_key(trades)

        key = ("Specific Asset(s)", ("BTC", "ETH"))
        assert key in grouped
        assert len(grouped[key]) == 2

    def test_empty_list(self):
        """Should return empty dict for empty trades list."""
        grouped = group_trades_by_key([])
        assert grouped == {}

    def test_single_trade(self, sample_trade_long):
        """Should handle single trade."""
        grouped = group_trades_by_key([sample_trade_long])

        assert len(grouped) == 1
        key = ("Specific Asset(s)", ("BTC", "ETH"))
        assert len(grouped[key]) == 1
