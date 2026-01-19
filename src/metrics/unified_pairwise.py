"""Unified pairwise agreement calculation - computes all cases in a single pass."""

import copy
from dataclasses import dataclass

import polars as pl

from cli.utils.annotation_model import Annotation
from src.agreement.unified import UnifiedAgreementResult, calculate_unified_agreement
from src.models.constants import AGREEMENT_FIELDS, ALL_LABEL_KEYS
from src.models.trade import normalize_annotations


def validate_and_dump_annotations(raw_annotations: list[dict] | None) -> list[dict]:
    """Validate annotations through Pydantic model and return as dicts."""
    if not raw_annotations:
        return []

    result = []
    for annot in raw_annotations:
        try:
            validated = Annotation.model_validate(annot)
            result.append(validated.model_dump())
        except Exception:
            continue
    return result


@dataclass
class AggregatedScores:
    """Aggregated scores for a single annotator pair across all tasks."""

    # Overall agreement (averaged across tasks)
    overall: float

    # Per-field scores (averaged across tasks)
    per_field: dict[str, float]

    # Per-label ratios (agreements / counts)
    per_label_ratios: dict[str, float]

    # Per-label raw agreement counts (for gt_counts output)
    per_label_counts: dict[str, float]

    # Metadata
    num_tasks: int


@dataclass
class AllPairScores:
    """All pairwise scores for a dataset."""

    # Nested dict: annotator1 -> annotator2 -> AggregatedScores
    scores: dict[str, dict[str, AggregatedScores | None]]

    # List of all annotators
    annotators: list[str]


class UnifiedPairwiseCalculator:
    """
    Calculates all agreement types between all annotator pairs in a single pass.

    This replaces PairwiseCalculator and computes:
    - Overall agreement (case=None)
    - Per-field agreement (case="field")
    - Per-label agreement (case="label")

    All three are computed from a single trade matching operation per task,
    reducing computation time by ~60-70%.
    """

    def __init__(self, common: bool = False):
        """
        Initialize unified pairwise calculator.

        Args:
            common: If True, only compare on commonly-annotated tasks
        """
        self.common = common

    def calculate_all_pairs(
        self,
        data: pl.DataFrame,
        annotators: list[str],
    ) -> AllPairScores:
        """
        Calculate all agreement types between all pairs of annotators.

        Returns AllPairScores containing aggregated scores for each pair.
        """
        scores: dict[str, dict[str, AggregatedScores | None]] = {}

        for annotator_1 in annotators:
            scores[annotator_1] = {}

            for annotator_2 in annotators:
                if annotator_1 == annotator_2:
                    scores[annotator_1][annotator_2] = None
                    continue

                scores[annotator_1][annotator_2] = self._calculate_pair(
                    data, annotator_1, annotator_2
                )

        return AllPairScores(scores=scores, annotators=annotators)

    def _calculate_pair(
        self,
        data: pl.DataFrame,
        annotator_1: str,
        annotator_2: str,
    ) -> AggregatedScores | None:
        """
        Calculate all agreement types between two annotators.

        Returns AggregatedScores or None if no common tasks.
        """
        # Filter to rows where annotator_1 has annotations
        temp = data.filter(pl.col(annotator_1).is_not_null())

        # Handle ground truth special case
        if "ground_truth" in [annotator_1, annotator_2] and not self.common:
            temp_2 = (
                temp.filter(pl.col(annotator_2).is_not_null())
                .filter(
                    ~pl.col("ground_truth_member").is_in([annotator_1, annotator_2])
                )
                .select([annotator_1, annotator_2])
            )
        else:
            temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
                [annotator_1, annotator_2]
            )

        if temp_2.shape[0] == 0:
            return None

        temp_2 = temp_2.to_dicts()

        # Collect results from all tasks
        task_results: list[UnifiedAgreementResult] = []

        for row in temp_2:
            # Validate through Pydantic model
            validated_1 = validate_and_dump_annotations(row[annotator_1])
            validated_2 = validate_and_dump_annotations(row[annotator_2])

            # Normalize annotations
            trades_1 = normalize_annotations(copy.deepcopy(validated_1))
            trades_2 = normalize_annotations(copy.deepcopy(validated_2))

            # Calculate ALL agreement types in a single pass
            result = calculate_unified_agreement(trades_1, trades_2)
            task_results.append(result)

        if not task_results:
            return None

        # Aggregate results across all tasks
        return self._aggregate_task_results(task_results)

    def _aggregate_task_results(
        self,
        task_results: list[UnifiedAgreementResult],
    ) -> AggregatedScores:
        """Aggregate results from multiple tasks into final scores."""
        n = len(task_results)

        # Overall agreement: simple average across tasks
        overall = sum(r.overall for r in task_results) / n

        # Per-field: average across tasks
        field_totals = {field: 0.0 for field in AGREEMENT_FIELDS}
        for result in task_results:
            for key, value in result.per_field.items():
                field_totals[key] += value
        per_field = {key: total / n for key, total in field_totals.items()}

        # Per-label: sum agreements and counts, then compute ratios
        total_agreements = {key: 0.0 for key in ALL_LABEL_KEYS}
        total_counts = {key: 0.0 for key in ALL_LABEL_KEYS}

        for result in task_results:
            for key, value in result.label_agreements.items():
                total_agreements[key] += value
            for key, value in result.label_counts.items():
                total_counts[key] += value

        # Compute ratios (agreements / counts)
        per_label_ratios = {
            key: total_agreements[key] / total_counts[key]
            if total_counts[key] > 0
            else 0.0
            for key in ALL_LABEL_KEYS
        }

        return AggregatedScores(
            overall=overall,
            per_field=per_field,
            per_label_ratios=per_label_ratios,
            per_label_counts=total_agreements,  # Raw counts for gt_counts output
            num_tasks=n,
        )
