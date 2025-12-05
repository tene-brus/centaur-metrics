"""Trade normalization utilities.

This module provides functions to normalize raw annotation dicts
and group trades by their primary key for agreement calculations.
"""

from src.models.constants import (
    EXPOSURE_CHANGE_FIELDS,
    OPTIONAL_FLAGS_FIELDS,
    POSITION_STATUS_FIELDS,
)


def normalize_position_status(annotations: list[dict]) -> list[dict]:
    """Normalize action_/state_position_status to unified position_status field."""
    for field in POSITION_STATUS_FIELDS:
        for annot in annotations:
            if field in annot:
                annot["position_status"] = annot[field]
                del annot[field]
    return annotations


def normalize_exposure_change(annotations: list[dict]) -> list[dict]:
    """Normalize action_/state_exposure_change to unified exposure_change field."""
    for field in EXPOSURE_CHANGE_FIELDS:
        for annot in annotations:
            if field in annot:
                annot["exposure_change"] = annot[field]
                del annot[field]
    return annotations


def normalize_optional_task_flags(annotations: list[dict]) -> list[dict]:
    """Normalize optional task flags and merge state_total_retro_flag."""
    for field in OPTIONAL_FLAGS_FIELDS:
        for annot in annotations:
            if field in annot:
                annot["optional_task_flags"] = annot[field] if annot[field] else []

                if field == "state_optional_task_flags":
                    if annot.get("state_total_retro_flag"):
                        annot["optional_task_flags"].append(
                            annot["state_total_retro_flag"]
                        )
                        del annot["state_total_retro_flag"]

                del annot[field]
    return annotations


def normalize_annotations(annotations: list[dict]) -> list[dict]:
    """
    Normalize a list of raw annotation dicts.

    Consolidates position_status, exposure_change, and optional_task_flags
    from their action_/state_ prefixed variants into unified field names.
    """
    annotations = normalize_position_status(annotations)
    annotations = normalize_exposure_change(annotations)
    annotations = normalize_optional_task_flags(annotations)
    return annotations


def get_primary_key(trade: dict) -> tuple:
    """
    Generate unique primary key for a trade based on asset reference.

    Trades are grouped by (asset_reference_type, specific_assets) for comparison.
    For "Specific Asset(s)", the specific_assets are sorted for consistent matching.
    """
    asset_ref_type = trade.get("asset_reference_type")
    specific_assets = trade.get("specific_assets")

    if asset_ref_type == "Specific Asset(s)":
        return (
            asset_ref_type,
            tuple(sorted(specific_assets)) if specific_assets else tuple(),
        )
    else:
        return (asset_ref_type, None)


def group_trades_by_key(trades: list[dict]) -> dict[tuple, list[dict]]:
    """Group trades by their primary key."""
    grouped: dict[tuple, list[dict]] = {}
    for trade in trades:
        key = get_primary_key(trade)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(trade)
    return grouped
