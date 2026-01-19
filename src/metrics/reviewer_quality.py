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
class VerifierOwnSubmissionStats:
    """Stats for tasks where verifier submitted their own annotation as GT."""

    total: int = 0
    # Reviewed by reviewer
    reviewed_total: int = 0
    reviewed_with_errors: int = 0  # Reviewer annotation != GT (reviewer mistake)
    # Not reviewed by reviewer
    not_reviewed_total: int = 0


@dataclass
class VerifierAcceptedStats:
    """Stats for tasks where a verifier accepted another annotator's submission as GT."""

    total: int = 0
    # Reviewed by reviewer
    reviewed_total: int = 0
    reviewed_with_errors: int = 0  # Reviewer annotation != GT (reviewer mistake)
    # Not reviewed by reviewer
    not_reviewed_total: int = 0


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
    tasks_not_reviewed: int = 0  # Tasks where reviewer has no annotation
    # Per-verifier stats (when verifier is ground_truth_member - verifier's own submission)
    verifier_own_submission_stats: dict[str, VerifierOwnSubmissionStats] | None = None
    # Per-verifier stats (when verifier accepted another annotator's submission as GT)
    verifier_accepted_stats: dict[str, VerifierAcceptedStats] | None = None


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


def _check_annotation_error(reviewer_ann, gt_ann) -> bool:
    """Check if reviewer annotation differs from GT (has error)."""
    if reviewer_ann is None:
        return False
    reviewer_validated = validate_and_dump_annotations(reviewer_ann)
    gt_validated = validate_and_dump_annotations(gt_ann)
    reviewer_trades = normalize_annotations(copy.deepcopy(reviewer_validated))
    gt_trades = normalize_annotations(copy.deepcopy(gt_validated))
    return not annotations_match(reviewer_trades, gt_trades)


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

    gt_verifiers = gt_verifiers or []

    # Add computed columns for analysis
    computed_cols = [
        pl.col("ground_truth").is_not_null().alias("has_gt"),
        pl.col(reviewer_email).is_not_null().alias("reviewer_reviewed"),
    ]

    # Handle optional columns
    if "ground_truth_member" in data.columns:
        computed_cols.append(
            pl.col("ground_truth_member").is_in(gt_verifiers).alias("gt_is_verifier")
        )
    else:
        computed_cols.append(pl.lit(False).alias("gt_is_verifier"))

    df = data.with_columns(computed_cols)

    # Add has_error column using map_elements (needs row-by-row check)
    df = df.with_columns(
        pl.struct([reviewer_email, "ground_truth"])
        .map_elements(
            lambda row: _check_annotation_error(
                row[reviewer_email], row["ground_truth"]
            ),
            return_dtype=pl.Boolean,
        )
        .alias("has_error")
    )

    # Project-level stats
    project_total_tasks = df.shape[0]

    # Tasks not reviewed (reviewer has no annotation but GT exists)
    tasks_not_reviewed = df.filter(
        ~pl.col("reviewer_reviewed") & pl.col("has_gt")
    ).shape[0]

    # Reviewed tasks with GT
    reviewed_with_gt = df.filter(pl.col("reviewer_reviewed") & pl.col("has_gt"))
    total_tasks = reviewed_with_gt.shape[0]
    tasks_with_errors = reviewed_with_gt.filter(pl.col("has_error")).shape[0]
    error_frequency = tasks_with_errors / total_tasks if total_tasks > 0 else 0.0

    # Per-trader breakdown (only reviewed tasks)
    per_trader: dict[str, dict] = {}
    if "trader" in df.columns:
        trader_stats = (
            reviewed_with_gt.group_by("trader")
            .agg(
                [
                    pl.len().alias("total"),
                    pl.col("has_error").sum().alias("errors"),
                ]
            )
            .to_dicts()
        )
        for row in trader_stats:
            trader = row["trader"] or "Unknown"
            total = row["total"]
            errors = row["errors"]
            per_trader[trader] = {
                "total": total,
                "errors": errors,
                "frequency": errors / total if total > 0 else 0.0,
            }
    elif total_tasks > 0:
        # No trader column - group all under "Unknown"
        per_trader["Unknown"] = {
            "total": total_tasks,
            "errors": tasks_with_errors,
            "frequency": error_frequency,
        }

    # Initialize verifier stats
    verifier_own_stats: dict[str, VerifierOwnSubmissionStats] = {}
    verifier_accepted_stats: dict[str, VerifierAcceptedStats] = {}
    for verifier in gt_verifiers:
        verifier_own_stats[verifier] = VerifierOwnSubmissionStats()
        verifier_accepted_stats[verifier] = VerifierAcceptedStats()

    # Verifier own submissions (ground_truth_member is a verifier)
    if gt_verifiers:
        own_sub_df = df.filter(pl.col("has_gt") & pl.col("gt_is_verifier"))
        for verifier in gt_verifiers:
            verifier_df = own_sub_df.filter(pl.col("ground_truth_member") == verifier)
            stats = verifier_own_stats[verifier]
            stats.total = verifier_df.shape[0]
            stats.reviewed_total = verifier_df.filter(
                pl.col("reviewer_reviewed")
            ).shape[0]
            stats.reviewed_with_errors = verifier_df.filter(
                pl.col("reviewer_reviewed") & pl.col("has_error")
            ).shape[0]
            stats.not_reviewed_total = verifier_df.filter(
                ~pl.col("reviewer_reviewed")
            ).shape[0]

    # Verifier accepted other annotator's submission (gt_accepted_by is set)
    if gt_verifiers and "gt_accepted_by" in df.columns:
        accepted_df = df.filter(
            pl.col("has_gt")
            & ~pl.col("gt_is_verifier")
            & pl.col("gt_accepted_by").is_not_null()
        )
        for verifier in gt_verifiers:
            verifier_df = accepted_df.filter(pl.col("gt_accepted_by") == verifier)
            stats = verifier_accepted_stats[verifier]
            stats.total = verifier_df.shape[0]
            stats.reviewed_total = verifier_df.filter(
                pl.col("reviewer_reviewed")
            ).shape[0]
            stats.reviewed_with_errors = verifier_df.filter(
                pl.col("reviewer_reviewed") & pl.col("has_error")
            ).shape[0]
            stats.not_reviewed_total = verifier_df.filter(
                ~pl.col("reviewer_reviewed")
            ).shape[0]

    return ReviewerErrorFrequency(
        project_name=project_name,
        reviewer_email=reviewer_email,
        total_tasks=total_tasks,
        tasks_with_errors=tasks_with_errors,
        error_frequency=error_frequency,
        per_trader=per_trader,
        project_total_tasks=project_total_tasks,
        tasks_not_reviewed=tasks_not_reviewed,
        verifier_own_submission_stats=verifier_own_stats
        if verifier_own_stats
        else None,
        verifier_accepted_stats=verifier_accepted_stats
        if verifier_accepted_stats
        else None,
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
