import itertools
import logging
from collections import defaultdict

from pydantic import BaseModel, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from cli.utils.valid_tickers import VALID_TICKERS

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Field names used in agreement calculations
AGREEMENT_FIELDS = [
    "direction",
    "position_status",
    "exposure_change",
    "state_type",
    "optional_task_flags",
]

# Valid values for each field
FIELD_VALUES = {
    "label_type": ["action", "state"],
    "asset_reference_type": [
        "Specific Asset(s)",
        "Majors",
        "DeFi",
        "AI",
        "AI Agents",
        "RWA",
        "Layer 1",
        "Layer 2",
        "Alts",
        "All Open Positions",
        "All Long Positions",
        "All Shorts",
        "Memes",
        "Other",
    ],
    "direction": ["Long", "Short", "Unclear"],
    "position_status": ["Clearly a new position", "Clearly an existing position"],
    "exposure_change": ["Increase", "Decrease", "Unclear", "No Change"],
    "remaining_exposure": ["Some", "None", "Unclear"],
    "state_type": ["Explicit State", "Direct State", "Indirect State"],
}

# Field normalization mappings
POSITION_STATUS = [
    "action_position_status",
    "state_position_status",
]

EXPOSURE_CHANGE = [
    "action_exposure_change",
    "state_exposure_change",
]

OPTIONAL_STATE_FLAGS = [
    "state_optional_task_flags",
    "action_optional_task_flags",
]

# Agreement scoring weights
PRIMARY_KEY_WEIGHT = 0.25  # Weight for matching primary key (asset reference)
REMAINING_FIELDS_WEIGHT = 0.75  # Weight for remaining fields after primary key match
PER_LABEL_BASE_SCORE = 0.05  # Base score for per-label calculations
SIMILARITY_FIELDS_COUNT = 5  # Number of fields used in similarity calculation


# ============================================================================
# ANNOTATION MODEL
# ============================================================================


class Annotation(BaseModel):
    label_type: str | None = Field(
        description="""
        ## Action vs. State (classification)
        - Action: explicit position change (open/increase/decrease/close). If action then action_exposure_change != No Change.
        - State: describes an existing position (e.g., “still holding”, “moved SL to BE”).
        If state then state_exposure_change can either equal to 'No Change' or 'Unclear'.
        """,
    )
    asset_reference_type: str | None = Field(default=MISSING)
    specific_assets: list[str] | None = Field(
        description=f"The ticker code of the detected assets. Valid tickers are {VALID_TICKERS}.",
        default=MISSING,
    )
    direction: str | None = Field(
        description="""
        In trading, there are two common positions: Long and Short.

        - A long position means you buy an asset (like a stock or crypto) because you believe its price will go up.
        If you buy a stock at $100 and sell it later at $120, you make a $20 profit.
        But if the price drops to $80 and you sell, you take a $20 loss. Long traders profit from rising prices.

        - A short position means you borrow an asset and sell it right away because you think its price will go down.
        For example, if you short a stock at $100 and it falls to $80, you buy it back cheaper and keep the $20 difference as profit.
        However, if the stock rises to $120, you’d lose $20. Short traders profit from falling prices, but they carry higher risk since the price could, in theory, rise infinitely.

        Think of it like this:
        Going long is like buying a house hoping its value will go up.
        Going short is like renting someone else’s house and selling it now, planning to buy it back later at a cheaper price.

        If you are not sure what is the direction of the message given, output 'Unclear'.
        """,
        default=MISSING,
    )

    action_position_status: str | None = Field(
        description="""
        Only valid values are either 'Clearly a new position' or 'Clearly an existing position'.
        Only if label_type is 'action'.
        """,
        default=MISSING,
    )
    state_position_status: str | None = Field(
        description="""
        Only valid values are either 'Clearly a new position' or 'Clearly an existing position'.
        Only if label_type is 'state'.
        """,
        default=MISSING,
    )

    action_exposure_change: str | None = Field(
        description="""
        The change in the trader's exposure to a certain asset if label_type is 'action'.
        """,
        default=MISSING,
    )
    state_exposure_change: str | None = Field(
        description="""
        The change in the trader's exposure to a certain asset if label_type is 'state'.
        """,
        default=MISSING,
    )

    remaining_exposure: str | None = Field(
        description="""
        If an exposure change was detected then how much is the remaining exposure.
        Valid values are ['Some', 'None', 'Unclear']
        """,
        default=MISSING,
    )

    state_type: str | None = Field(
        description="""
        How clearly does this message confirm that the trader is still holding a position? Must be filled for state trades.
        Valid values "Explicit State" | "Direct State" | "Indirect State". Only for state trades.
        """,
        default=MISSING,
    )

    state_optional_task_flags: list[str] | None = Field(default=MISSING)
    action_optional_task_flags: list[str] | None = Field(default=MISSING)

    state_total_retro_flag: str | None = Field(default=MISSING)

    @model_validator(mode="after")
    def _check_optional_flags(self):
        if self.label_type == "action":
            delattr(self, "state_optional_task_flags")
        elif self.label_type == "state":
            delattr(self, "action_optional_task_flags")

        return self

    @model_validator(mode="after")
    def _validate_fields(self):
        """Consolidated validator for all field value checks."""
        validation_rules = {
            "label_type": ["state", "action"],
            "remaining_exposure": ["Some", "None", "Unclear"],
            "asset_reference_type": FIELD_VALUES["asset_reference_type"],
            "direction": ["Long", "Short", "Unclear"],
            "action_exposure_change": ["Increase", "Decrease", "Unclear"],
            "state_exposure_change": ["No Change", "Unclear"],
            "action_position_status": [
                "Clearly a new position",
                "Clearly an existing position",
            ],
            "state_position_status": [
                "Clearly a new position",
                "Clearly an existing position",
            ],
            "state_type": ["Explicit State", "Direct State", "Indirect State"],
        }

        for field_name, valid_values in validation_rules.items():
            self._check_function(field_name, valid_values)

        return self

    def _check_function(self, attribute: str, valid_values: list):
        variable = getattr(self, attribute)
        if variable != MISSING and variable is not None:
            if variable not in valid_values:
                logger.error(f"'{attribute}' equals {variable}")
                raise ValueError(
                    f"'{attribute}' should either be equal to one of {valid_values}"
                )


# ============================================================================
# LIST ANNOTATIONS MODEL
# ============================================================================


class ListAnnotations(BaseModel):
    annotations: list[Annotation] = []

    # ------------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------------

    def agreement(self, annotation: type[BaseModel], case: str | None):
        if case:
            agreement = self._eq_per_label(annotation, case=case)
        else:
            agreement = self._eq(annotation, case=case)

        return agreement

    # ------------------------------------------------------------------------
    # Helper Methods - Normalization and Grouping
    # ------------------------------------------------------------------------

    def _normalize_annotations(self, annotations: list[dict]) -> list[dict]:
        """Normalize annotations by consolidating position_status, exposure_change, and optional_task_flags."""
        annotations = normalize_position_status(annotations)
        annotations = normalize_exposure_change(annotations)
        annotations = normalize_optional_task_flags(annotations)
        return annotations

    def _group_trades_by_key(self, trades: list[dict]) -> dict[tuple, list[dict]]:
        """Group trades by their primary key (asset_reference_type + specific_assets)."""
        grouped_trades = {}
        for trade in trades:
            key = self._get_primary_trade_key(trade)
            if key not in grouped_trades:
                grouped_trades[key] = []
            grouped_trades[key].append(trade)
        return grouped_trades

    # ------------------------------------------------------------------------
    # Agreement Calculation Methods
    # ------------------------------------------------------------------------

    def _eq(self, annot_2: list[dict], case: str | None):
        if not isinstance(annot_2, ListAnnotations):
            raise TypeError("Both objects should be of type ListAnnotations.")

        obj_1 = self._normalize_annotations(self.model_dump()["annotations"])
        obj_2 = self._normalize_annotations(annot_2.model_dump()["annotations"])

        # quick resolve of most cases
        if obj_1 == [] and obj_2 == []:
            return 1.0
        elif obj_1 == [] and obj_2 != []:
            return 0.0
        elif obj_1 != [] and obj_2 == []:
            return 0.0

        grouped_trades_1 = self._group_trades_by_key(obj_1)
        grouped_trades_2 = self._group_trades_by_key(obj_2)

        logger.info(f"Grouped Trades 1: {grouped_trades_1}")
        logger.info(f"Grouped Trades 2: {grouped_trades_2}")

        total_accumulated_agreement_score = 0.0

        # Get all unique primary keys identified by either annotator
        all_primary_keys = set(grouped_trades_1.keys()).union(
            set(grouped_trades_2.keys())
        )

        for primary_key in all_primary_keys:
            trades_A_for_key = grouped_trades_1.get(primary_key, [])
            trades_B_for_key = grouped_trades_2.get(primary_key, [])

            logger.info(f"Processing primary key group: {primary_key}")
            logger.info(f"  Trades A in group: {trades_A_for_key}")
            logger.info(f"  Trades B in group: {trades_B_for_key}")

            # Find best matching pairs within this group
            matches = self._find_best_matches(
                trades_A_for_key, trades_B_for_key, per_label=case
            )

            for trade_A, trade_B, similarity_score in matches:
                # Each matched trade contributes:
                # PRIMARY_KEY_WEIGHT (for the primary key match, as we are in this common group)
                # PLUS REMAINING_FIELDS_WEIGHT * (similarity_score) for the remaining fields
                single_trade_pair_total_score = PRIMARY_KEY_WEIGHT + (
                    REMAINING_FIELDS_WEIGHT * similarity_score
                )
                total_accumulated_agreement_score += single_trade_pair_total_score
                logger.info(
                    f"Matched pair in group {primary_key}: Score = {single_trade_pair_total_score}"
                )

        # Normalization: Divide by the maximum number of total trades found by either annotator
        max_possible_trades_count = max(len(obj_1), len(obj_2))

        agreement_score = total_accumulated_agreement_score / max_possible_trades_count

        logger.info(
            f"Total accumulated agreement score from matched pairs: {total_accumulated_agreement_score}"
        )
        logger.info(f"Final Agreement Score for this pair: {agreement_score}")

        return agreement_score

    def _eq_per_label(self, annot_2: list[dict], case: str | None):
        if not isinstance(annot_2, ListAnnotations):
            raise TypeError("Both objects should be of type ListAnnotations.")

        obj_1 = self._normalize_annotations(self.model_dump()["annotations"])
        obj_2 = self._normalize_annotations(annot_2.model_dump()["annotations"])

        # quick resolve of most cases
        if obj_1 == [] and obj_2 == []:
            if case == "label":
                return ({}, {})
            elif case == "field":
                return {field: 0.2 for field in AGREEMENT_FIELDS}
        elif obj_1 == [] and obj_2 != []:
            if case == "label":
                return (
                    {item: 0 for _, values in FIELD_VALUES.items() for item in values},
                    {item: 0 for _, values in FIELD_VALUES.items() for item in values},
                )
            elif case == "field":
                return {field: 0 for field in AGREEMENT_FIELDS}
        elif obj_1 != [] and obj_2 == []:
            if case == "label":
                return (
                    {item: 0 for _, values in FIELD_VALUES.items() for item in values},
                    {item: 0 for _, values in FIELD_VALUES.items() for item in values},
                )
            elif case == "field":
                return {field: 0 for field in AGREEMENT_FIELDS}

        grouped_trades_1 = self._group_trades_by_key(obj_1)
        grouped_trades_2 = self._group_trades_by_key(obj_2)

        logger.info(f"Grouped Trades 1: {grouped_trades_1}")
        logger.info(f"Grouped Trades 2: {grouped_trades_2}")

        # Get all unique primary keys identified by either annotator
        all_primary_keys = set(grouped_trades_1.keys()).union(
            set(grouped_trades_2.keys())
        )

        per_label_scores = []
        per_label_counts = []

        for primary_key in all_primary_keys:
            trades_A_for_key = grouped_trades_1.get(primary_key, [])
            trades_B_for_key = grouped_trades_2.get(primary_key, [])

            logger.info(f"Processing primary key group: {primary_key}")
            logger.info(f"  Trades A in group: {trades_A_for_key}")
            logger.info(f"  Trades B in group: {trades_B_for_key}")

            # Find best matching pairs within this group
            matches = self._find_best_matches(
                trades_A_for_key, trades_B_for_key, per_label=case
            )

            for trade_A, trade_B, similarity_score in matches:
                if case == "field":
                    score = {
                        key: PER_LABEL_BASE_SCORE + (REMAINING_FIELDS_WEIGHT * value)
                        for key, value in similarity_score[0].items()
                    }
                    per_label_scores.append(score)
                elif case == "label":
                    # similarity_score is (agreement_per_label, count_per_label, score)
                    per_label_scores.append(similarity_score[0])
                    per_label_counts.append(similarity_score[1])

        if case == "field":
            total = defaultdict(float)
            for item in per_label_scores:
                for key, value in item.items():
                    total[key] += value
            return {key: total[key] / len(per_label_scores) for key in total}
        elif case == "label":
            total_agreements = defaultdict(float)
            total_counts = defaultdict(float)
            for item in per_label_scores:
                for key, value in item.items():
                    total_agreements[key] += value
            for item in per_label_counts:
                for key, value in item.items():
                    total_counts[key] += value
            return (dict(total_agreements), dict(total_counts))

    # ------------------------------------------------------------------------
    # Similarity Calculation Methods
    # ------------------------------------------------------------------------

    def _calculate_annot_similarity_per_label(
        self, annot_1: dict, annot_2: dict
    ) -> float:
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
            if annot_1.get(field):
                count_per_label[annot_1.get(field)] += 1
            if annot_2.get(field) and annot_2.get(field) != annot_1.get(field):
                count_per_label[annot_2.get(field)] += 1
            # Track agreement
            if annot_1.get(field):
                if annot_1.get(field) == annot_2.get(field):
                    agreement_per_label[annot_1.get(field)] += 1

        nom = 0
        denom = 0
        for _, value in agreement_per_label.items():
            denom += 1
            nom += value

        return (agreement_per_label, count_per_label, nom / denom)

    def _calculate_annot_similarity(self, annot_1: dict, annot_2: dict) -> float:
        """
        Calculate similarity score between two trades based on
        position_status, direction, and exposure_change and optional_task_flags.
        optional_task_flags are averaged before being added to the overall score.
        Returns a value between 0 and 1.
        """
        score = 0
        denom = SIMILARITY_FIELDS_COUNT

        if annot_1.get("state_type") == annot_2.get("state_type"):
            score += 1

        if annot_1.get("direction") == annot_2.get("direction"):
            score += 1

        if annot_1.get("exposure_change") == annot_2.get("exposure_change"):
            score += 1

        if annot_1.get("position_status") == annot_2.get("position_status"):
            score += 1

        if annot_1.get("optional_task_flags") and annot_2.get("optional_task_flags"):
            temp_denom = max(
                len(annot_1.get("optional_task_flags")),
                len(annot_2.get("optional_task_flags")),
            )
            temp_nom = len(
                set(annot_1.get("optional_task_flags")).intersection(
                    set(annot_2.get("optional_task_flags"))
                )
            )
            score += temp_nom / temp_denom
        elif not annot_1.get("optional_task_flags") and not annot_2.get(
            "optional_task_flags"
        ):
            score += 1

        # calculate mean score
        return score / denom

    def _calculate_annot_similarity_per_field(
        self, annot_1: dict, annot_2: dict
    ) -> float:
        temp = {
            "state_type": 0,
            "direction": 0,
            "exposure_change": 0,
            "position_status": 0,
            "optional_task_flags": 0,
        }

        if annot_1.get("state_type") == annot_2.get("state_type"):
            temp["state_type"] += 1 / SIMILARITY_FIELDS_COUNT

        if annot_1.get("direction") == annot_2.get("direction"):
            temp["direction"] += 1 / SIMILARITY_FIELDS_COUNT

        if annot_1.get("exposure_change") == annot_2.get("exposure_change"):
            temp["exposure_change"] += 1 / SIMILARITY_FIELDS_COUNT

        if annot_1.get("position_status") == annot_2.get("position_status"):
            temp["position_status"] += 1 / SIMILARITY_FIELDS_COUNT

        if annot_1.get("optional_task_flags") and annot_2.get("optional_task_flags"):
            temp_denom = max(
                len(annot_1.get("optional_task_flags")),
                len(annot_2.get("optional_task_flags")),
            )
            temp_nom = len(
                set(annot_1.get("optional_task_flags")).intersection(
                    set(annot_2.get("optional_task_flags"))
                )
            )
            temp["optional_task_flags"] += (
                temp_nom / temp_denom
            ) / SIMILARITY_FIELDS_COUNT
        elif not annot_1.get("optional_task_flags") and not annot_2.get(
            "optional_task_flags"
        ):
            temp["optional_task_flags"] += 1 / SIMILARITY_FIELDS_COUNT

        nom = 0
        denom = 0
        for _, value in temp.items():
            denom += 1
            nom += value

        # calculate mean score
        return (temp, nom / denom)

    # ------------------------------------------------------------------------
    # Trade Matching Methods
    # ------------------------------------------------------------------------

    def _get_primary_trade_key(self, trade: dict) -> tuple:
        """Generates a unique primary key for a trade based on asset_reference_type and taxonomy, order-independent."""
        asset_ref_type = trade.get("asset_reference_type")
        specific_taxonomy = trade.get("specific_assets")
        if asset_ref_type == "Specific Asset(s)":
            # Ensure taxonomy is a tuple for hashing and comparison, order-independent
            return (
                asset_ref_type,
                tuple(sorted(specific_taxonomy)) if specific_taxonomy else tuple(),
            )
        else:
            return (asset_ref_type, None)

    def _find_best_matches(
        self, trades_A: list[dict], trades_B: list[dict], per_label: str | None
    ) -> list[tuple[dict, dict, float]]:
        """Find best matching pairs of trades using a greedy approach."""
        if not trades_A or not trades_B:
            return []

        # Calculate similarity matrix
        similarity_matrix = []
        for trade_A in trades_A:
            row = []
            for trade_B in trades_B:
                if per_label == "label":
                    similarity = self._calculate_annot_similarity_per_label(
                        trade_A, trade_B
                    )
                elif per_label == "field":
                    similarity = self._calculate_annot_similarity_per_field(
                        trade_A, trade_B
                    )
                else:
                    similarity = self._calculate_annot_similarity(trade_A, trade_B)
                row.append(similarity)
            similarity_matrix.append(row)

        # Find best matches greedily
        matches = []
        used_A = set()
        used_B = set()

        # Sort all possible pairs by similarity score
        all_pairs = []
        for i, trade_A in enumerate(trades_A):
            for j, trade_B in enumerate(trades_B):
                all_pairs.append((i, j, similarity_matrix[i][j]))

        # Sort by similarity score in descending order
        # For per_label="label", tuple is (agreement, count, score) so index [2]
        # For per_label="field", tuple is (field_scores, score) so index [1]
        all_pairs.sort(key=lambda x: x[2][2] if per_label == "label" else (x[2][1] if per_label == "field" else x[2]), reverse=True)

        # Greedily match pairs
        for i, j, score in all_pairs:
            if i not in used_A and j not in used_B:
                matches.append((trades_A[i], trades_B[j], score))
                used_A.add(i)
                used_B.add(j)

        return matches


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def normalize_position_status(annotation: dict):
    for field in POSITION_STATUS:
        for annot in annotation:
            if field in annot:
                annot["position_status"] = annot[field]
                del annot[field]

    return annotation


def normalize_exposure_change(annotation: dict):
    for field in EXPOSURE_CHANGE:
        for annot in annotation:
            if field in annot:
                annot["exposure_change"] = annot[field]
                del annot[field]

    return annotation


def normalize_optional_task_flags(annotation: dict):
    for field in OPTIONAL_STATE_FLAGS:
        for annot in annotation:
            if field in annot:
                annot["optional_task_flags"] = annot[field] if annot[field] else []

                if field == "state_optional_task_flags":
                    if annot.get("state_total_retro_flag"):
                        annot["optional_task_flags"].append(
                            annot["state_total_retro_flag"]
                        )

                        del annot["state_total_retro_flag"]

                del annot[field]

    return annotation
