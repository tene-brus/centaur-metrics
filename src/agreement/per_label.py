"""Per-label agreement calculator - produces agreement and count dicts per label value."""

from src.agreement.base import AgreementCalculator
from src.agreement.matching import match_trades_by_group
from src.models.constants import FIELD_VALUES

# All possible label values across all fields
ALL_LABEL_VALUES = [item for values in FIELD_VALUES.values() for item in values]


class PerLabelAgreementCalculator(AgreementCalculator):
    """
    Calculates agreement broken down by label value.

    For each possible label value (e.g., "Long", "Short", "Increase"),
    tracks how often annotators agreed when that label was used.

    Returns both agreement scores and raw counts for each label.
    """

    def calculate(
        self, trades_a: list[dict], trades_b: list[dict]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Calculate per-label agreement between two normalized trade lists."""
        # Handle empty cases
        if trades_a == [] and trades_b == []:
            return ({}, {})
        elif trades_a == [] and trades_b != []:
            zero_dict = {item: 0 for _, values in FIELD_VALUES.items() for item in values}
            return (zero_dict.copy(), zero_dict.copy())
        elif trades_a != [] and trades_b == []:
            zero_dict = {item: 0 for _, values in FIELD_VALUES.items() for item in values}
            return (zero_dict.copy(), zero_dict.copy())

        # Find matching trade pairs
        matches = match_trades_by_group(trades_a, trades_b, self.similarity)

        # Accumulate per-label scores and counts
        per_label_scores = []
        per_label_counts = []

        for trade_a, trade_b, similarity_info in matches:
            agreement_dict, count_dict, _ = similarity_info
            per_label_scores.append(agreement_dict)
            per_label_counts.append(count_dict)

        # Sum up agreements and counts
        total_agreements = {item: 0.0 for _, values in FIELD_VALUES.items() for item in values}
        total_counts = {item: 0.0 for _, values in FIELD_VALUES.items() for item in values}

        for item in per_label_scores:
            for key, value in item.items():
                total_agreements[key] += value

        for item in per_label_counts:
            for key, value in item.items():
                total_counts[key] += value

        return (total_agreements, total_counts)

    def similarity(
        self, trade_a: dict, trade_b: dict
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """
        Calculate per-label similarity between two trades.

        Returns a tuple of (agreement_per_label, count_per_label, overall_score).
        """
        fields = [
            "label_type",
            "asset_reference_type",
            "direction",
            "position_status",
            "exposure_change",
            "remaining_exposure",
            "state_type",
        ]

        agreement_per_label = {
            item: 0 for _, values in FIELD_VALUES.items() for item in values
        }
        count_per_label = {
            item: 0 for _, values in FIELD_VALUES.items() for item in values
        }

        for field in fields:
            # Track count for any label that either annotator submitted
            if trade_a.get(field):
                count_per_label[trade_a.get(field)] += 1
            if trade_b.get(field) and trade_b.get(field) != trade_a.get(field):
                count_per_label[trade_b.get(field)] += 1
            # Track agreement
            if trade_a.get(field):
                if trade_a.get(field) == trade_b.get(field):
                    agreement_per_label[trade_a.get(field)] += 1

        nom = 0
        denom = 0
        for _, value in agreement_per_label.items():
            denom += 1
            nom += value

        return (agreement_per_label, count_per_label, nom / denom)
