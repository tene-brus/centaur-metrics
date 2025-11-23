import itertools
import logging
from collections import defaultdict

from pydantic import BaseModel, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from cli.utils.valid_tickers import VALID_TICKERS

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
    def _check_extract_trade(self):
        valid_values = ["state", "action"]
        self._check_function("label_type", valid_values)

        return self

    @model_validator(mode="after")
    def _check_remaining_exposure(self):
        valid_values = ["Some", "None", "Unclear"]
        self._check_function("remaining_exposure", valid_values)

        return self

    @model_validator(mode="after")
    def _check_asset_reference(self):
        valid_values = [
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
        ]
        self._check_function("asset_reference_type", valid_values)

        return self

    @model_validator(mode="after")
    def _check_direction(self):
        valid_values = ["Long", "Short", "Unclear"]
        self._check_function("direction", valid_values)

        return self

    @model_validator(mode="after")
    def _check_exposure_change(self):
        valid_values = ["Increase", "Decrease", "Unclear"]
        self._check_function("action_exposure_change", valid_values)

        valid_values = ["No Change", "Unclear"]
        self._check_function("state_exposure_change", valid_values)

        return self

    @model_validator(mode="after")
    def _check_position_status(self):
        valid_values = ["Clearly a new position", "Clearly an existing position"]
        self._check_function("action_position_status", valid_values)
        self._check_function("state_position_status", valid_values)

        return self

    @model_validator(mode="after")
    def _check_state_type(self):
        valid_values = ["Explicit State", "Direct State", "Indirect State"]
        self._check_function("state_type", valid_values)

        return self

    def _check_function(self, attribute: str, valid_values: list):
        variable = getattr(self, attribute)
        if variable != MISSING and variable is not None:
            if variable not in valid_values:
                logger.error(f"'{attribute}' equals {variable}")
                raise ValueError(
                    f"'{attribute}' should either be equal to one of {valid_values}"
                )


class ListAnnotations(BaseModel):
    annotations: list[Annotation] = []

    def agreement(self, annotation: type[BaseModel], per_label=False):
        if per_label:
            agreement = self._eq_per_label(annotation)
        else:
            agreement = self._eq(annotation)

        return agreement

    def _gen_permutations(self, annot_1: list[dict], annot_2: list[dict]):
        # rule to avoid very huge permutation loops
        if len(annot_1) >= 9:
            annot_1 = annot_1[:8]

        if len(annot_2) >= 9:
            annot_2 = annot_2[:8]

        larger, smaller = annot_1, annot_2
        flip = False
        if len(annot_1) < len(annot_2):
            larger, smaller = annot_2, annot_1
            flip = True

        permutations = []
        for subset in itertools.permutations(larger, len(smaller)):
            if flip:
                # if we swapped them, flip the pairs back
                pairs = list(zip(smaller, subset))
            else:
                pairs = list(zip(subset, smaller))

            permutations.append(pairs)

        return permutations

    def _eq(self, annot_2: list[dict]):
        if not isinstance(annot_2, ListAnnotations):
            raise TypeError("Both objects should be of type ListAnnotations.")

        obj_1: list[dict] = self.model_dump()["annotations"]
        obj_1 = normalize_position_status(obj_1)
        obj_1 = normalize_exposure_change(obj_1)
        obj_1 = normalize_optional_task_flags(obj_1)

        obj_2: list[dict] = annot_2.model_dump()["annotations"]
        obj_2 = normalize_position_status(obj_2)
        obj_2 = normalize_exposure_change(obj_2)
        obj_2 = normalize_optional_task_flags(obj_2)

        # quick resolve of most cases
        if obj_1 == [] and obj_2 == []:
            return 1.0
        elif obj_1 == [] and obj_2 != []:
            return 0.0
        elif obj_1 != [] and obj_2 == []:
            return 0.0

        # permutations = self._gen_permutations(obj_1, obj_2)

        # Group trades by their primary key (asset_reference_type + specific_assets)
        grouped_trades_1: dict[tuple, list[dict]] = {}
        for trade in obj_1:
            key = self._get_primary_trade_key(trade)
            if key not in grouped_trades_1:
                grouped_trades_1[key] = []
            grouped_trades_1[key].append(trade)

        grouped_trades_2: dict[tuple, list[dict]] = {}
        for trade in obj_2:
            key = self._get_primary_trade_key(trade)
            if key not in grouped_trades_2:
                grouped_trades_2[key] = []
            grouped_trades_2[key].append(trade)

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
            matches = self._find_best_matches(trades_A_for_key, trades_B_for_key)

            for trade_A, trade_B, similarity_score in matches:
                # Each matched trade contributes:
                # 0.25 (for the primary key match, as we are in this common group)
                # PLUS 0.75 * (similarity_score) for the remaining 3 fields
                single_trade_pair_total_score = 0.25 + (0.75 * similarity_score)
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

    def _eq_per_label(self, annot_2: list[dict]):
        if not isinstance(annot_2, ListAnnotations):
            raise TypeError("Both objects should be of type ListAnnotations.")

        obj_1: list[dict] = self.model_dump()["annotations"]
        obj_1 = normalize_position_status(obj_1)
        obj_1 = normalize_exposure_change(obj_1)
        obj_1 = normalize_optional_task_flags(obj_1)

        obj_2: list[dict] = annot_2.model_dump()["annotations"]
        obj_2 = normalize_position_status(obj_2)
        obj_2 = normalize_exposure_change(obj_2)
        obj_2 = normalize_optional_task_flags(obj_2)

        # quick resolve of most cases
        if obj_1 == [] and obj_2 == []:
            return {}
        elif obj_1 == [] and obj_2 != []:
            return {item: 0 for _, values in FIELD_VALUES.items() for item in values}
        elif obj_1 != [] and obj_2 == []:
            return {item: 0 for _, values in FIELD_VALUES.items() for item in values}

        # Group trades by their primary key (asset_reference_type + specific_assets)
        grouped_trades_1: dict[tuple, list[dict]] = {}
        for trade in obj_1:
            key = self._get_primary_trade_key(trade)
            if key not in grouped_trades_1:
                grouped_trades_1[key] = []
            grouped_trades_1[key].append(trade)

        grouped_trades_2: dict[tuple, list[dict]] = {}
        for trade in obj_2:
            key = self._get_primary_trade_key(trade)
            if key not in grouped_trades_2:
                grouped_trades_2[key] = []
            grouped_trades_2[key].append(trade)

        logger.info(f"Grouped Trades 1: {grouped_trades_1}")
        logger.info(f"Grouped Trades 2: {grouped_trades_2}")

        # Get all unique primary keys identified by either annotator
        all_primary_keys = set(grouped_trades_1.keys()).union(
            set(grouped_trades_2.keys())
        )

        per_label_scores = []

        for primary_key in all_primary_keys:
            trades_A_for_key = grouped_trades_1.get(primary_key, [])
            trades_B_for_key = grouped_trades_2.get(primary_key, [])

            logger.info(f"Processing primary key group: {primary_key}")
            logger.info(f"  Trades A in group: {trades_A_for_key}")
            logger.info(f"  Trades B in group: {trades_B_for_key}")

            # Find best matching pairs within this group
            matches = self._find_best_matches(
                trades_A_for_key, trades_B_for_key, per_label=True
            )

            for trade_A, trade_B, similarity_score in matches:
                ##### PER LABEL WORKS UNTIL HERE, MODIFICATIONS NEEDED AFTER THIS STEP
                # single_trade_pair_total_score = 0.25 + (0.75 * similarity_score)
                per_label_scores.append(similarity_score[0])

        total_per_label_scores = defaultdict(int)
        for item in per_label_scores:
            for key in item:
                total_per_label_scores[key] += item[key]
        # Normalization: Divide by the maximum number of total trades found by either annotator
        # max_possible_trades_count = max(len(obj_1), len(obj_2))

        return total_per_label_scores

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
        aggreement_per_label = {
            item: 0 for _, values in FIELD_VALUES.items() for item in values
        }
        for field in fields:
            if annot_1.get(field):
                if annot_1.get(field) == annot_2.get(field):
                    aggreement_per_label[annot_1.get(field)] += 1

        nom = 0
        denom = 0
        for _, value in aggreement_per_label.items():
            denom += 1
            nom += value

        return (aggreement_per_label, nom / denom)

    def _calculate_annot_similarity(self, annot_1: dict, annot_2: dict) -> float:
        """
        Calculate similarity score between two trades based on
        position_status, direction, and exposure_change and optional_task_flags.
        optional_task_flags are averaged before being added to the overall score.
        Returns a value between 0 and 1.
        """
        score = 0
        denom = 5

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
        self, trades_A: list[dict], trades_B: list[dict], per_label: bool = False
    ) -> list[tuple[dict, dict, float]]:
        """Find best matching pairs of trades using a greedy approach."""
        if not trades_A or not trades_B:
            return []

        # Calculate similarity matrix
        similarity_matrix = []
        for trade_A in trades_A:
            row = []
            for trade_B in trades_B:
                if per_label:
                    similarity = self._calculate_annot_similarity_per_label(
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
        all_pairs.sort(key=lambda x: x[2][1] if per_label else x[2], reverse=True)

        # Greedily match pairs
        for i, j, score in all_pairs:
            if i not in used_A and j not in used_B:
                matches.append((trades_A[i], trades_B[j], score))
                used_A.add(i)
                used_B.add(j)

        return matches


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
