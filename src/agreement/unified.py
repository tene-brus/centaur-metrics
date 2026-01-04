"""Unified agreement calculator - computes all agreement types in a single pass."""

from dataclasses import dataclass

from src.agreement.matching import match_trades_by_group
from src.models.constants import (
    AGREEMENT_FIELDS,
    ALL_LABEL_KEYS,
    PER_LABEL_BASE_SCORE,
    PRIMARY_KEY_WEIGHT,
    REMAINING_FIELDS_WEIGHT,
    SIMILARITY_FIELDS_COUNT,
    get_label_key,
)


@dataclass
class UnifiedSimilarity:
    """Result of unified similarity calculation for a single trade pair."""

    # Overall agreement: single float (0-1)
    overall_score: float

    # Per-field agreement: dict mapping field -> normalized score (0-0.2 per field)
    field_scores: dict[str, float]

    # Per-label agreement: dicts for agreements and counts
    label_agreements: dict[str, int]
    label_counts: dict[str, int]


@dataclass
class UnifiedAgreementResult:
    """Complete agreement result for a pair of trade lists (one task)."""

    # Overall agreement score for this task
    overall: float

    # Per-field scores (weighted and averaged across matches)
    per_field: dict[str, float]

    # Per-label raw agreements and counts (summed across matches)
    label_agreements: dict[str, float]
    label_counts: dict[str, float]

    # Metadata
    num_matches: int
    trades_a_count: int
    trades_b_count: int


def unified_similarity(trade_a: dict, trade_b: dict) -> UnifiedSimilarity:
    """
    Calculate all similarity metrics for a single trade pair in one pass.

    This replaces the three separate similarity functions:
    - OverallAgreementCalculator.similarity()
    - PerFieldAgreementCalculator.similarity()
    - PerLabelAgreementCalculator.similarity()
    """
    # Initialize results
    field_scores = {
        "state_type": 0.0,
        "direction": 0.0,
        "exposure_change": 0.0,
        "position_status": 0.0,
        "optional_task_flags": 0.0,
    }
    label_agreements = {key: 0 for key in ALL_LABEL_KEYS}
    label_counts = {key: 0 for key in ALL_LABEL_KEYS}

    overall_score = 0.0

    # === Compare the 5 core similarity fields ===
    # These are used for overall and per_field calculations

    if trade_a.get("state_type") == trade_b.get("state_type"):
        overall_score += 1
        field_scores["state_type"] = 1 / SIMILARITY_FIELDS_COUNT

    if trade_a.get("direction") == trade_b.get("direction"):
        overall_score += 1
        field_scores["direction"] = 1 / SIMILARITY_FIELDS_COUNT

    if trade_a.get("exposure_change") == trade_b.get("exposure_change"):
        overall_score += 1
        field_scores["exposure_change"] = 1 / SIMILARITY_FIELDS_COUNT

    if trade_a.get("position_status") == trade_b.get("position_status"):
        overall_score += 1
        field_scores["position_status"] = 1 / SIMILARITY_FIELDS_COUNT

    # Handle optional_task_flags specially
    flags_a = trade_a.get("optional_task_flags")
    flags_b = trade_b.get("optional_task_flags")
    if flags_a and flags_b:
        temp_denom = max(len(flags_a), len(flags_b))
        temp_nom = len(set(flags_a).intersection(set(flags_b)))
        flag_score = temp_nom / temp_denom
        overall_score += flag_score
        field_scores["optional_task_flags"] = flag_score / SIMILARITY_FIELDS_COUNT
    elif not flags_a and not flags_b:
        overall_score += 1
        field_scores["optional_task_flags"] = 1 / SIMILARITY_FIELDS_COUNT

    # Normalize overall score
    overall_score = overall_score / SIMILARITY_FIELDS_COUNT

    # === Compare fields for per-label tracking ===
    # These 7 fields are used for per_label calculations
    label_fields = [
        "label_type",
        "asset_reference_type",
        "direction",
        "position_status",
        "exposure_change",
        "remaining_exposure",
        "state_type",
    ]

    for field in label_fields:
        label_a = trade_a.get(field)
        label_b = trade_b.get(field)

        # Track count for any label that either annotator submitted
        if label_a:
            key_a = get_label_key(label_a, field)
            label_counts[key_a] += 1

        if label_b and label_b != label_a:
            key_b = get_label_key(label_b, field)
            label_counts[key_b] += 1

        # Track agreement
        if label_a and label_a == label_b:
            key = get_label_key(label_a, field)
            label_agreements[key] += 1

    return UnifiedSimilarity(
        overall_score=overall_score,
        field_scores=field_scores,
        label_agreements=label_agreements,
        label_counts=label_counts,
    )


def _extract_unified_score(similarity: UnifiedSimilarity) -> float:
    """Extract numeric score from UnifiedSimilarity for sorting during matching."""
    return similarity.overall_score


def calculate_unified_agreement(
    trades_a: list[dict],
    trades_b: list[dict],
) -> UnifiedAgreementResult:
    """
    Calculate all agreement types between two trade lists in a single pass.

    This replaces:
    - OverallAgreementCalculator.calculate()
    - PerFieldAgreementCalculator.calculate()
    - PerLabelAgreementCalculator.calculate()

    The trade matching happens only ONCE, and all metrics are extracted from
    the same matched pairs.
    """
    # Handle empty cases
    if trades_a == [] and trades_b == []:
        return UnifiedAgreementResult(
            overall=1.0,
            per_field={field: 0.2 for field in AGREEMENT_FIELDS},
            label_agreements={},
            label_counts={},
            num_matches=0,
            trades_a_count=0,
            trades_b_count=0,
        )
    elif trades_a == [] and trades_b != []:
        return UnifiedAgreementResult(
            overall=0.0,
            per_field={field: 0 for field in AGREEMENT_FIELDS},
            label_agreements={key: 0 for key in ALL_LABEL_KEYS},
            label_counts={key: 0 for key in ALL_LABEL_KEYS},
            num_matches=0,
            trades_a_count=0,
            trades_b_count=len(trades_b),
        )
    elif trades_a != [] and trades_b == []:
        return UnifiedAgreementResult(
            overall=0.0,
            per_field={field: 0 for field in AGREEMENT_FIELDS},
            label_agreements={key: 0 for key in ALL_LABEL_KEYS},
            label_counts={key: 0 for key in ALL_LABEL_KEYS},
            num_matches=0,
            trades_a_count=len(trades_a),
            trades_b_count=0,
        )

    # === SINGLE PASS: Match trades and compute all similarities ===
    matches = match_trades_by_group(trades_a, trades_b, unified_similarity)

    # === Extract metrics from matches ===

    # Overall agreement
    total_overall_score = 0.0
    for trade_a, trade_b, similarity in matches:
        pair_score = PRIMARY_KEY_WEIGHT + (
            REMAINING_FIELDS_WEIGHT * similarity.overall_score
        )
        total_overall_score += pair_score

    max_trades = max(len(trades_a), len(trades_b))
    overall = total_overall_score / max_trades

    # Per-field agreement
    if matches:
        field_totals = {field: 0.0 for field in AGREEMENT_FIELDS}
        for trade_a, trade_b, similarity in matches:
            for key, value in similarity.field_scores.items():
                # Apply weighting: PER_LABEL_BASE_SCORE + (REMAINING_FIELDS_WEIGHT * value)
                weighted_value = PER_LABEL_BASE_SCORE + (
                    REMAINING_FIELDS_WEIGHT * value
                )
                field_totals[key] += weighted_value

        per_field = {key: total / len(matches) for key, total in field_totals.items()}
    else:
        per_field = {field: 0.0 for field in AGREEMENT_FIELDS}

    # Per-label agreements and counts
    total_label_agreements = {key: 0.0 for key in ALL_LABEL_KEYS}
    total_label_counts = {key: 0.0 for key in ALL_LABEL_KEYS}

    for trade_a, trade_b, similarity in matches:
        for key, value in similarity.label_agreements.items():
            total_label_agreements[key] += value
        for key, value in similarity.label_counts.items():
            total_label_counts[key] += value

    return UnifiedAgreementResult(
        overall=overall,
        per_field=per_field,
        label_agreements=total_label_agreements,
        label_counts=total_label_counts,
        num_matches=len(matches),
        trades_a_count=len(trades_a),
        trades_b_count=len(trades_b),
    )
