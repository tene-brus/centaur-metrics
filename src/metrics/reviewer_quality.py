"""Calculate reviewer error frequency - how often reviewer annotations differ from GT.

Error frequency = tasks where reviewer != GT / total reviewer tasks
"""

import copy
import os
from dataclasses import dataclass

import polars as pl

from src.metrics.unified_pairwise import validate_and_dump_annotations
from src.models.trade import normalize_annotations


@dataclass
class ReviewerErrorFrequency:
    """Result of reviewer error frequency analysis."""

    project_name: str
    reviewer_email: str
    total_tasks: int
    tasks_with_errors: int
    error_frequency: float  # tasks_with_errors / total_tasks
    per_trader: dict[str, dict]  # trader -> {total, errors, frequency}
    # Project-level statistics
    project_total_tasks: int = 0  # Total tasks in the project
    gt_verifier_stats: dict[str, dict] = (
        None  # GT verifier -> {total_verified, reviewed_by_reviewer}
    )


def annotations_match(trades_a: list[dict], trades_b: list[dict]) -> bool:
    """Check if two sets of normalized trades are identical.

    Returns True if they match completely, False if there's any difference.
    """
    if len(trades_a) != len(trades_b):
        return False

    # Sort trades by a consistent key for comparison
    def trade_key(t):
        return (
            t.get("asset_reference_type", ""),
            str(sorted(t.get("specific_assets", []) or [])),
            t.get("label_type", ""),
            t.get("direction", ""),
            t.get("exposure_change", ""),
            t.get("position_status", ""),
        )

    sorted_a = sorted(trades_a, key=trade_key)
    sorted_b = sorted(trades_b, key=trade_key)

    for ta, tb in zip(sorted_a, sorted_b):
        if trade_key(ta) != trade_key(tb):
            return False

    return True


def calculate_reviewer_error_frequency(
    data: pl.DataFrame,
    reviewer_email: str,
    project_name: str = "",
    gt_verifiers: list[str] | None = None,
) -> ReviewerErrorFrequency | None:
    """Calculate reviewer error frequency.

    Error frequency = tasks where reviewer annotations != GT / total tasks

    Args:
        data: Polars DataFrame with annotations
        reviewer_email: Email of the reviewer to analyze
        project_name: Name of the project (for reporting)
        gt_verifiers: List of GT verifier emails to track stats for

    Returns:
        ReviewerErrorFrequency with error counts and frequency
    """
    if reviewer_email not in data.columns:
        return None

    if "ground_truth" not in data.columns:
        return None

    # Calculate project-level statistics
    project_total_tasks = data.shape[0]

    # Calculate GT verifier statistics
    gt_verifier_stats: dict[str, dict] = {}
    if gt_verifiers and "ground_truth_member" in data.columns:
        for verifier in gt_verifiers:
            # Count tasks where this verifier provided GT
            tasks_verified = data.filter(
                pl.col("ground_truth_member") == verifier
            ).shape[0]

            # Count tasks where this verifier provided GT AND reviewer reviewed
            tasks_reviewed_by_reviewer = data.filter(
                (pl.col("ground_truth_member") == verifier)
                & pl.col(reviewer_email).is_not_null()
                & pl.col("ground_truth").is_not_null()
            ).shape[0]

            gt_verifier_stats[verifier] = {
                "total_verified": tasks_verified,
                "reviewed_by_reviewer": tasks_reviewed_by_reviewer,
            }

    # Filter to rows where both reviewer and GT exist
    filtered = data.filter(
        pl.col(reviewer_email).is_not_null() & pl.col("ground_truth").is_not_null()
    )

    if filtered.shape[0] == 0:
        return None

    total_tasks = 0
    tasks_with_errors = 0
    per_trader: dict[str, dict] = {}

    for row in filtered.to_dicts():
        trader = row.get("trader", "Unknown")

        # Initialize trader tracking
        if trader not in per_trader:
            per_trader[trader] = {"total": 0, "errors": 0}

        # Get and normalize annotations
        reviewer_raw = row[reviewer_email]
        gt_raw = row["ground_truth"]

        reviewer_validated = validate_and_dump_annotations(reviewer_raw)
        gt_validated = validate_and_dump_annotations(gt_raw)

        reviewer_trades = normalize_annotations(copy.deepcopy(reviewer_validated))
        gt_trades = normalize_annotations(copy.deepcopy(gt_validated))

        # Check if they match
        total_tasks += 1
        per_trader[trader]["total"] += 1

        if not annotations_match(reviewer_trades, gt_trades):
            tasks_with_errors += 1
            per_trader[trader]["errors"] += 1

    # Calculate frequencies
    error_frequency = tasks_with_errors / total_tasks if total_tasks > 0 else 0.0

    for trader_data in per_trader.values():
        trader_data["frequency"] = (
            trader_data["errors"] / trader_data["total"]
            if trader_data["total"] > 0
            else 0.0
        )

    return ReviewerErrorFrequency(
        project_name=project_name,
        reviewer_email=reviewer_email,
        total_tasks=total_tasks,
        tasks_with_errors=tasks_with_errors,
        error_frequency=error_frequency,
        per_trader=per_trader,
        project_total_tasks=project_total_tasks,
        gt_verifier_stats=gt_verifier_stats if gt_verifier_stats else None,
    )


def calculate_reviewer_error_frequency_from_file(
    data_path: str,
    reviewer_email: str,
) -> ReviewerErrorFrequency | None:
    """Calculate reviewer error frequency from a JSONL file.

    Args:
        data_path: Path to the JSONL file
        reviewer_email: Email of the reviewer to analyze

    Returns:
        ReviewerErrorFrequency with error counts and frequency
    """
    project_name = os.path.splitext(os.path.basename(data_path))[0]
    data = pl.read_ndjson(data_path, infer_schema_length=8000)
    return calculate_reviewer_error_frequency(data, reviewer_email, project_name)
