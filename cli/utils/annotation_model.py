import itertools
import logging

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from cli.utils.valid_tickers import VALID_TICKERS

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        agreement = self.__eq__(annotation, per_label=per_label)
        return agreement

    def __eq__(self, value, per_label=False):
        if not isinstance(value, ListAnnotations):
            raise TypeError("Both object sould be of type ListAnnotations.")

        obj_1: list[dict] = self.model_dump()["annotations"]
        obj_1 = normalize_position_status(obj_1)
        obj_1 = normalize_exposure_change(obj_1)
        obj_1 = normalize_optional_task_flags(obj_1)

        obj_2: list[dict] = value.model_dump()["annotations"]
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

        # rule to avoid very huge permutation loops
        if len(obj_1) >= 9:
            obj_1 = obj_1[:8]

        if len(obj_2) >= 9:
            obj_2 = obj_2[:8]

        larger, smaller = obj_1, obj_2
        flip = False
        if len(obj_1) < len(obj_2):
            larger, smaller = obj_2, obj_1
            flip = True

        permutations = []
        for subset in itertools.permutations(larger, len(smaller)):
            if flip:
                # if we swapped them, flip the pairs back
                pairs = list(zip(smaller, subset))
            else:
                pairs = list(zip(subset, smaller))

            permutations.append(pairs)

        perm_scores = []
        for perm in permutations:
            scores = []
            for annot_1, annot_2 in perm:
                if not per_label:
                    perm_score = self._calculate_annot_similarity(annot_1, annot_2)
                else:
                    perm_score = self._calculate_annot_similarity_per_label(
                        annot_1, annot_2
                    )
                scores.append(perm_score)

            # Normalization: Divide by the maximum number of total trades found by either annotator
            denom = max(len(obj_1), len(obj_2))
            perm_scores.append(np.sum(scores).item() / denom)

            # if the new addition is equal to 1 then return this score
            if perm_scores[-1] == 1:
                return perm_scores[-1]

        max_idx = np.argmax(perm_scores)

        return perm_scores[max_idx]

    def _calculate_annot_similarity_per_label(
        self, annot_1: dict, annot_2: dict
    ) -> float:
        pass

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
                len(annot_1.get("optional_task_flags")),
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
                annot["optional_task_flags"] = annot[field]

                if field == "state_optional_task_flags":
                    if annot["state_total_retro_flag"]:
                        annot["optional_task_flags"].append(
                            annot["state_total_retro_flag"]
                        )

                    del annot["state_total_retro_flag"]

                del annot[field]

    return annotation
