"""Pairwise agreement calculation between all annotators."""

import copy
from collections import defaultdict
from typing import Literal

import polars as pl

from src.agreement.overall import OverallAgreementCalculator
from src.agreement.per_field import PerFieldAgreementCalculator
from src.agreement.per_label import PerLabelAgreementCalculator
from src.models.trade import normalize_annotations

# Import Annotation model for validation (filters out invalid fields via validators)
from cli.utils.annotation_model import Annotation


def validate_and_dump_annotations(raw_annotations: list[dict] | None) -> list[dict]:
    """
    Validate annotations through Pydantic model and return as dicts.

    This ensures the same validation/filtering as the original ListAnnotations.
    """
    if not raw_annotations:
        return []

    result = []
    for annot in raw_annotations:
        try:
            validated = Annotation.model_validate(annot)
            result.append(validated.model_dump())
        except Exception:
            # Skip invalid annotations
            continue
    return result


class PairwiseCalculator:
    """
    Calculates pairwise agreement between all annotators.

    Uses the refactored agreement calculators for computation.
    """

    def __init__(
        self,
        case: Literal["field", "label"] | None = None,
        common: bool = False,
    ):
        """
        Initialize pairwise calculator.

        Args:
            case: Type of agreement calculation (None=overall, "field", "label")
            common: If True, only compare on commonly-annotated tasks
        """
        self.case = case
        self.common = common

        # Select the appropriate calculator
        if case == "field":
            self.calculator = PerFieldAgreementCalculator()
        elif case == "label":
            self.calculator = PerLabelAgreementCalculator()
        else:
            self.calculator = OverallAgreementCalculator()

    @classmethod
    def create(
        cls,
        case: Literal["field", "label"] | None = None,
        common: bool = False,
    ) -> "PairwiseCalculator":
        """Factory method to create pairwise calculator."""
        return cls(case=case, common=common)

    def calculate_pair(
        self,
        data: pl.DataFrame,
        annotator_1: str,
        annotator_2: str,
    ) -> list | float | None:
        """
        Calculate agreement between two annotators.

        Returns the same types as the original calculate_pairwise_agreement:
        - float for case=None
        - list for case="field" or case="label"
        - None if no common tasks
        """
        # Filter to rows where annotator_1 has annotations
        temp = data.filter(pl.col(annotator_1).is_not_null())

        if self.case:
            agreement_scores = []
        else:
            agreement_scores = 0.0

        # Handle ground truth special case
        if "ground_truth" in [annotator_1, annotator_2] and not self.common:
            temp_2 = (
                temp.filter(pl.col(annotator_2).is_not_null())
                .filter(~pl.col("ground_truth_member").is_in([annotator_1, annotator_2]))
                .select([annotator_1, annotator_2])
            )
        else:
            temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
                [annotator_1, annotator_2]
            )

        if temp_2.shape[0] == 0:
            return None if self.case is None else []

        temp_2 = temp_2.to_dicts()
        denom = 0

        for row in temp_2:
            # Validate through Pydantic model (same as original ListAnnotations)
            validated_1 = validate_and_dump_annotations(row[annotator_1])
            validated_2 = validate_and_dump_annotations(row[annotator_2])

            # Normalize annotations (deep copy to avoid mutation issues)
            trades_1 = normalize_annotations(copy.deepcopy(validated_1))
            trades_2 = normalize_annotations(copy.deepcopy(validated_2))

            if self.case == "label":
                # Skip rows where both annotators have empty annotations
                if trades_1 == [] and trades_2 == []:
                    continue
                result = self.calculator.calculate(trades_1, trades_2)
                agreement_scores.append(result)
                denom += 1
            elif self.case == "field":
                result = self.calculator.calculate(trades_1, trades_2)
                agreement_scores.append(result)
                denom += 1
            else:
                result = self.calculator.calculate(trades_1, trades_2)
                agreement_scores += result
                denom += 1

        if denom == 0:
            return None if self.case is None else []

        if self.case is None:
            agreement_scores = agreement_scores / denom

        return agreement_scores

    def calculate_all_pairs(
        self,
        data: pl.DataFrame,
        annotators: list[str],
    ) -> dict[str, dict[str, list | float | None]]:
        """
        Calculate agreement between all pairs of annotators.

        Returns a nested dict: annotator1 -> annotator2 -> score(s)
        """
        scores: dict[str, dict[str, list | float | None]] = {}

        for annotator_1 in annotators:
            scores[annotator_1] = {}

            for annotator_2 in annotators:
                if annotator_1 == annotator_2:
                    scores[annotator_1][annotator_2] = None
                    continue

                scores[annotator_1][annotator_2] = self.calculate_pair(
                    data, annotator_1, annotator_2
                )

        return scores

    def aggregate_per_label_scores(
        self,
        scores: dict[str, dict[str, list | None]],
        annotators: list[str],
        average: bool = True,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Aggregate per-label or per-field scores into mean values or total counts.

        This matches the original aggregate_per_label_scores function.
        """
        result: dict[str, dict[str, dict[str, float]]] = {}

        for annotator in annotators:
            result[annotator] = {}

            for annotator_2 in annotators:
                if annotator_2 == annotator:
                    continue

                score_list = scores[annotator][annotator_2]
                if score_list is None or len(score_list) == 0:
                    result[annotator][annotator_2] = {}
                    continue

                if self.case == "label":
                    # For label case, scores are tuples of (agreements_dict, counts_dict)
                    agreements: dict[str, float] = defaultdict(float)
                    counts: dict[str, float] = defaultdict(float)

                    for item in score_list:
                        agreements_dict, counts_dict = item
                        for key, value in agreements_dict.items():
                            agreements[key] += value
                        for key, value in counts_dict.items():
                            counts[key] += value

                    if average:
                        # Compute ratio: agreements / counts (percentage)
                        result[annotator][annotator_2] = {
                            key: agreements[key] / counts[key] if counts[key] > 0 else 0.0
                            for key in agreements.keys()
                        }
                    else:
                        # Return raw agreement counts
                        result[annotator][annotator_2] = dict(agreements)
                else:
                    # For field case, scores are dicts
                    field_totals: dict[str, float] = defaultdict(float)
                    n = len(score_list)

                    for item in score_list:
                        for key, value in item.items():
                            field_totals[key] += value

                    if average:
                        result[annotator][annotator_2] = {
                            key: total / n
                            for key, total in field_totals.items()
                        }
                    else:
                        result[annotator][annotator_2] = dict(field_totals)

        return result
