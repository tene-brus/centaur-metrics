"""Per-label agreement calculator - produces agreement and count dicts per label value."""

from src.agreement.base import AgreementCalculator
from src.agreement.matching import match_trades_by_group
from src.models.constants import AMBIGUOUS_LABELS, FIELD_VALUES


def get_label_key(label: str, field: str) -> str:
    """Get the key for a label, adding field context for ambiguous labels."""
    if label in AMBIGUOUS_LABELS:
        return f"{label} ({field})"
    return label


def get_all_label_keys() -> list[str]:
    """Get all possible label keys, with field-specific keys for ambiguous labels."""
    keys = []
    for field, values in FIELD_VALUES.items():
        for label in values:
            key = get_label_key(label, field)
            if key not in keys:
                keys.append(key)
    return keys


# All possible label keys (with field disambiguation for ambiguous labels)
ALL_LABEL_KEYS = get_all_label_keys()


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
            zero_dict = {key: 0 for key in ALL_LABEL_KEYS}
            return (zero_dict.copy(), zero_dict.copy())
        elif trades_a != [] and trades_b == []:
            zero_dict = {key: 0 for key in ALL_LABEL_KEYS}
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
        total_agreements = {key: 0.0 for key in ALL_LABEL_KEYS}
        total_counts = {key: 0.0 for key in ALL_LABEL_KEYS}

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
        Uses field-specific keys for ambiguous labels (e.g., "Unclear (direction)").
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

        agreement_per_label = {key: 0 for key in ALL_LABEL_KEYS}
        count_per_label = {key: 0 for key in ALL_LABEL_KEYS}

        for field in fields:
            # Track count for any label that either annotator submitted
            label_a = trade_a.get(field)
            label_b = trade_b.get(field)

            if label_a:
                key_a = get_label_key(label_a, field)
                count_per_label[key_a] += 1

            if label_b and label_b != label_a:
                key_b = get_label_key(label_b, field)
                count_per_label[key_b] += 1

            # Track agreement
            if label_a and label_a == label_b:
                key = get_label_key(label_a, field)
                agreement_per_label[key] += 1

        nom = 0
        denom = 0
        for _, value in agreement_per_label.items():
            denom += 1
            nom += value

        return (agreement_per_label, count_per_label, nom / denom)
