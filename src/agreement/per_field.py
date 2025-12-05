"""Per-field agreement calculator - produces a dict of field -> score."""

from src.agreement.base import AgreementCalculator
from src.agreement.matching import match_trades_by_group
from src.models.constants import (
    AGREEMENT_FIELDS,
    PER_LABEL_BASE_SCORE,
    REMAINING_FIELDS_WEIGHT,
    SIMILARITY_FIELDS_COUNT,
)


class PerFieldAgreementCalculator(AgreementCalculator):
    """
    Calculates agreement broken down by field.

    Returns a dict mapping each field name to its agreement score,
    allowing analysis of which fields have highest/lowest agreement.
    """

    def calculate(self, trades_a: list[dict], trades_b: list[dict]) -> dict[str, float]:
        """Calculate per-field agreement between two normalized trade lists."""
        # Handle empty cases
        if trades_a == [] and trades_b == []:
            return {field: 0.2 for field in AGREEMENT_FIELDS}
        elif trades_a == [] and trades_b != []:
            return {field: 0 for field in AGREEMENT_FIELDS}
        elif trades_a != [] and trades_b == []:
            return {field: 0 for field in AGREEMENT_FIELDS}

        # Find matching trade pairs
        matches = match_trades_by_group(trades_a, trades_b, self.similarity)

        # Accumulate per-field scores with weighting
        per_label_scores = []
        for trade_a, trade_b, similarity_info in matches:
            field_scores, _ = similarity_info
            # Apply weighting: PER_LABEL_BASE_SCORE + (REMAINING_FIELDS_WEIGHT * value)
            weighted_scores = {
                key: PER_LABEL_BASE_SCORE + (REMAINING_FIELDS_WEIGHT * value)
                for key, value in field_scores.items()
            }
            per_label_scores.append(weighted_scores)

        # Average across all matches
        if per_label_scores:
            total = {field: 0.0 for field in AGREEMENT_FIELDS}
            for item in per_label_scores:
                for key, value in item.items():
                    total[key] += value
            return {key: total[key] / len(per_label_scores) for key in total}
        else:
            return {field: 0.0 for field in AGREEMENT_FIELDS}

    def similarity(self, trade_a: dict, trade_b: dict) -> tuple[dict[str, float], float]:
        """
        Calculate per-field similarity between two trades.

        Returns a tuple of (field_scores_dict, overall_score).
        """
        temp = {
            "state_type": 0,
            "direction": 0,
            "exposure_change": 0,
            "position_status": 0,
            "optional_task_flags": 0,
        }

        if trade_a.get("state_type") == trade_b.get("state_type"):
            temp["state_type"] += 1 / SIMILARITY_FIELDS_COUNT

        if trade_a.get("direction") == trade_b.get("direction"):
            temp["direction"] += 1 / SIMILARITY_FIELDS_COUNT

        if trade_a.get("exposure_change") == trade_b.get("exposure_change"):
            temp["exposure_change"] += 1 / SIMILARITY_FIELDS_COUNT

        if trade_a.get("position_status") == trade_b.get("position_status"):
            temp["position_status"] += 1 / SIMILARITY_FIELDS_COUNT

        if trade_a.get("optional_task_flags") and trade_b.get("optional_task_flags"):
            temp_denom = max(
                len(trade_a.get("optional_task_flags")),
                len(trade_b.get("optional_task_flags")),
            )
            temp_nom = len(
                set(trade_a.get("optional_task_flags")).intersection(
                    set(trade_b.get("optional_task_flags"))
                )
            )
            temp["optional_task_flags"] += (temp_nom / temp_denom) / SIMILARITY_FIELDS_COUNT
        elif not trade_a.get("optional_task_flags") and not trade_b.get("optional_task_flags"):
            temp["optional_task_flags"] += 1 / SIMILARITY_FIELDS_COUNT

        nom = 0
        denom = 0
        for _, value in temp.items():
            denom += 1
            nom += value

        return (temp, nom / denom)
