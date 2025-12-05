"""Constants for annotation agreement calculations."""

from cli.utils.valid_tickers import VALID_TICKERS

# ============================================================================
# AGREEMENT SCORING WEIGHTS
# ============================================================================

PRIMARY_KEY_WEIGHT = 0.25  # Weight for matching primary key (asset reference)
REMAINING_FIELDS_WEIGHT = 0.75  # Weight for remaining fields after primary key match
PER_LABEL_BASE_SCORE = 0.05  # Base score for per-label calculations
SIMILARITY_FIELDS_COUNT = 5  # Number of fields used in similarity calculation


# ============================================================================
# FIELD DEFINITIONS
# ============================================================================

AGREEMENT_FIELDS = [
    "direction",
    "position_status",
    "exposure_change",
    "state_type",
    "optional_task_flags",
]

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

# Column names for output CSVs
LABEL_COLUMNS = [
    "action",
    "state",
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
    "Long",
    "Short",
    "Unclear",
    "Clearly a new position",
    "Clearly an existing position",
    "Increase",
    "Decrease",
    "No Change",
    "Some",
    "None",
    "Explicit State",
    "Direct State",
    "Indirect State",
]

FIELD_COLUMNS = [
    "state_type",
    "direction",
    "exposure_change",
    "position_status",
    "optional_task_flags",
]


# ============================================================================
# FIELD NORMALIZATION MAPPINGS
# ============================================================================

POSITION_STATUS_FIELDS = [
    "action_position_status",
    "state_position_status",
]

EXPOSURE_CHANGE_FIELDS = [
    "action_exposure_change",
    "state_exposure_change",
]

OPTIONAL_FLAGS_FIELDS = [
    "state_optional_task_flags",
    "action_optional_task_flags",
]


# ============================================================================
# VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    "label_type": ["state", "action"],
    "remaining_exposure": ["Some", "None", "Unclear"],
    "asset_reference_type": FIELD_VALUES["asset_reference_type"],
    "direction": ["Long", "Short", "Unclear"],
    "action_exposure_change": ["Increase", "Decrease", "Unclear"],
    "state_exposure_change": ["No Change", "Unclear"],
    "action_position_status": ["Clearly a new position", "Clearly an existing position"],
    "state_position_status": ["Clearly a new position", "Clearly an existing position"],
    "state_type": ["Explicit State", "Direct State", "Indirect State"],
}

# Re-export for convenience
__all__ = [
    "VALID_TICKERS",
    "PRIMARY_KEY_WEIGHT",
    "REMAINING_FIELDS_WEIGHT",
    "PER_LABEL_BASE_SCORE",
    "SIMILARITY_FIELDS_COUNT",
    "AGREEMENT_FIELDS",
    "FIELD_VALUES",
    "LABEL_COLUMNS",
    "FIELD_COLUMNS",
    "POSITION_STATUS_FIELDS",
    "EXPOSURE_CHANGE_FIELDS",
    "OPTIONAL_FLAGS_FIELDS",
    "VALIDATION_RULES",
]
