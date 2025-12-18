"""Shared pytest fixtures for all tests."""

import json

import polars as pl
import pytest


# ============================================================================
# SAMPLE TRADE DATA
# ============================================================================


@pytest.fixture
def sample_trade_long():
    """Sample trade annotation with Long direction."""
    return {
        "label_type": "action",
        "asset_reference_type": "Specific Asset(s)",
        "specific_assets": ["BTC", "ETH"],
        "direction": "Long",
        "position_status": "Clearly a new position",
        "exposure_change": "Increase",
        "remaining_exposure": "Some",
        "state_type": None,
        "optional_task_flags": [],
    }


@pytest.fixture
def sample_trade_short():
    """Sample trade annotation with Short direction."""
    return {
        "label_type": "action",
        "asset_reference_type": "Specific Asset(s)",
        "specific_assets": ["BTC", "ETH"],
        "direction": "Short",
        "position_status": "Clearly an existing position",
        "exposure_change": "Decrease",
        "remaining_exposure": "None",
        "state_type": None,
        "optional_task_flags": [],
    }


@pytest.fixture
def sample_trade_state():
    """Sample state-type trade annotation."""
    return {
        "label_type": "state",
        "asset_reference_type": "Majors",
        "specific_assets": None,
        "direction": "Long",
        "position_status": "Clearly an existing position",
        "exposure_change": "No Change",
        "remaining_exposure": "Some",
        "state_type": "Explicit State",
        "optional_task_flags": ["flag1"],
    }


@pytest.fixture
def sample_trade_unclear():
    """Sample trade with Unclear values."""
    return {
        "label_type": "action",
        "asset_reference_type": "Alts",
        "specific_assets": None,
        "direction": "Unclear",
        "position_status": "Clearly a new position",
        "exposure_change": "Unclear",
        "remaining_exposure": "Unclear",
        "state_type": None,
        "optional_task_flags": [],
    }


@pytest.fixture
def sample_raw_annotation_action():
    """Raw annotation with action_* prefixed fields (pre-normalization)."""
    return {
        "label_type": "action",
        "asset_reference_type": "Specific Asset(s)",
        "specific_assets": ["BTC"],
        "direction": "Long",
        "action_position_status": "Clearly a new position",
        "action_exposure_change": "Increase",
        "remaining_exposure": "Some",
        "action_optional_task_flags": ["flag1", "flag2"],
    }


@pytest.fixture
def sample_raw_annotation_state():
    """Raw annotation with state_* prefixed fields (pre-normalization)."""
    return {
        "label_type": "state",
        "asset_reference_type": "Majors",
        "specific_assets": None,
        "direction": "Long",
        "state_position_status": "Clearly an existing position",
        "state_exposure_change": "No Change",
        "remaining_exposure": "Some",
        "state_type": "Direct State",
        "state_optional_task_flags": ["flag1"],
        "state_total_retro_flag": "retro_flag",
    }


# ============================================================================
# SAMPLE JSONL DATA
# ============================================================================


@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content for testing DataLoader."""
    return [
        {
            "task_id": "task_1",
            "trader": "trader_A",
            "num_annotations": 2,
            "predictions": [{"direction": "Long"}],
            "ground_truth_member": None,
            "user1@example.com": [
                {
                    "label_type": "action",
                    "asset_reference_type": "Specific Asset(s)",
                    "specific_assets": ["BTC"],
                    "direction": "Long",
                    "action_position_status": "Clearly a new position",
                    "action_exposure_change": "Increase",
                }
            ],
            "user2@example.com": [
                {
                    "label_type": "action",
                    "asset_reference_type": "Specific Asset(s)",
                    "specific_assets": ["BTC"],
                    "direction": "Short",
                    "action_position_status": "Clearly a new position",
                    "action_exposure_change": "Decrease",
                }
            ],
        },
        {
            "task_id": "task_2",
            "trader": "trader_A",
            "num_annotations": 2,
            "predictions": [{"direction": "Short"}],
            "ground_truth_member": "user1@example.com",
            "ground_truth": [{"direction": "Long"}],
            "user1@example.com": [
                {
                    "label_type": "action",
                    "asset_reference_type": "Majors",
                    "direction": "Long",
                    "action_position_status": "Clearly an existing position",
                    "action_exposure_change": "Decrease",
                }
            ],
            "user2@example.com": [
                {
                    "label_type": "action",
                    "asset_reference_type": "Majors",
                    "direction": "Long",
                    "action_position_status": "Clearly an existing position",
                    "action_exposure_change": "Decrease",
                }
            ],
        },
        {
            "task_id": "task_3",
            "trader": "trader_B",
            "num_annotations": 1,
            "predictions": [{"direction": "Long"}],
            "ground_truth_member": None,
            "user1@example.com": [
                {
                    "label_type": "state",
                    "asset_reference_type": "DeFi",
                    "direction": "Long",
                    "state_position_status": "Clearly a new position",
                    "state_exposure_change": "No Change",
                    "state_type": "Explicit State",
                }
            ],
            "user2@example.com": None,
        },
        {
            "task_id": "task_4",
            "trader": "trader_B",
            "num_annotations": 0,  # Should be filtered out
            "predictions": None,
            "user1@example.com": None,
            "user2@example.com": None,
        },
    ]


@pytest.fixture
def temp_jsonl_file(sample_jsonl_content, tmp_path):
    """Create a temporary JSONL file with sample data."""
    jsonl_path = tmp_path / "test_data.jsonl"
    with open(jsonl_path, "w") as f:
        for item in sample_jsonl_content:
            f.write(json.dumps(item) + "\n")
    return str(jsonl_path)


# ============================================================================
# SAMPLE DATAFRAMES
# ============================================================================


@pytest.fixture
def sample_agreement_df():
    """Sample DataFrame for CSV utilities testing."""
    return pl.DataFrame(
        {
            "primary_annotator": ["user1", "user2", "user1", "user2"],
            "secondary_annotator": ["user2", "user1", "user2", "user1"],
            "trader": ["trader_A", "trader_A", "trader_B", "trader_B"],
            "common_tasks": [10, 10, 5, 5],
            "prim_annot_tasks": [15, 12, 8, 6],
            "direction": [0.8, 0.8, 0.9, 0.9],
            "exposure_change": [0.7, 0.7, 0.85, 0.85],
        }
    )


@pytest.fixture
def sample_gt_counts_df():
    """Sample DataFrame for GT counts testing."""
    return pl.DataFrame(
        {
            "primary_annotator": ["user1", "user2"],
            "secondary_annotator": ["ground_truth", "ground_truth"],
            "trader": ["trader_A", "trader_A"],
            "common_tasks": [20, 15],
            "prim_annot_tasks": [25, 20],
            "Long": [15, 10],
            "Short": [5, 5],
        }
    )
