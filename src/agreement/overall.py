"""Overall agreement calculator - produces a single float score."""

from src.agreement.base import AgreementCalculator
from src.agreement.matching import match_trades_by_group
from src.models.constants import (
    PRIMARY_KEY_WEIGHT,
    REMAINING_FIELDS_WEIGHT,
    SIMILARITY_FIELDS_COUNT,
)


class OverallAgreementCalculator(AgreementCalculator):
    """
    Calculates overall agreement as a single float score.

    The score is computed as:
    - For each matched trade pair: PRIMARY_KEY_WEIGHT + (REMAINING_FIELDS_WEIGHT * field_similarity)
    - Normalized by the maximum number of trades from either annotator
    """

    def calculate(self, trades_a: list[dict], trades_b: list[dict]) -> float:
        """Calculate overall agreement between two normalized trade lists."""
        # Handle empty cases
        if trades_a == [] and trades_b == []:
            return 1.0
        elif trades_a == [] and trades_b != []:
            return 0.0
        elif trades_a != [] and trades_b == []:
            return 0.0

        # Find matching trade pairs across all primary key groups
        matches = match_trades_by_group(trades_a, trades_b, self.similarity)

        # Accumulate agreement score
        total_score = 0.0
        for trade_a, trade_b, similarity_score in matches:
            # Each matched trade contributes:
            # PRIMARY_KEY_WEIGHT (for matching on same asset)
            # + REMAINING_FIELDS_WEIGHT * similarity_score (for field agreement)
            pair_score = PRIMARY_KEY_WEIGHT + (REMAINING_FIELDS_WEIGHT * similarity_score)
            total_score += pair_score

        # Normalize by max possible trades
        max_trades = max(len(trades_a), len(trades_b))
        return total_score / max_trades

    def similarity(self, trade_a: dict, trade_b: dict) -> float:
        """
        Calculate similarity score between two trades.

        Compares 5 fields: state_type, direction, exposure_change, position_status, optional_task_flags.
        Returns a value between 0 and 1.
        """
        score = 0
        denom = SIMILARITY_FIELDS_COUNT

        if trade_a.get("state_type") == trade_b.get("state_type"):
            score += 1

        if trade_a.get("direction") == trade_b.get("direction"):
            score += 1

        if trade_a.get("exposure_change") == trade_b.get("exposure_change"):
            score += 1

        if trade_a.get("position_status") == trade_b.get("position_status"):
            score += 1

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
            score += temp_nom / temp_denom
        elif not trade_a.get("optional_task_flags") and not trade_b.get("optional_task_flags"):
            score += 1

        return score / denom
