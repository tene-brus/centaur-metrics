"""Tests for src/models/constants.py."""

import pytest

from src.models.constants import (
    AGREEMENT_FIELDS,
    AMBIGUOUS_LABELS,
    FIELD_COLUMNS,
    FIELD_VALUES,
    LABEL_COLUMNS,
    PRIMARY_KEY_WEIGHT,
    REMAINING_FIELDS_WEIGHT,
    VALIDATION_RULES,
)


class TestAgreementWeights:
    """Tests for agreement scoring weights."""

    def test_weights_sum_to_one(self):
        """PRIMARY_KEY_WEIGHT + REMAINING_FIELDS_WEIGHT should equal 1."""
        assert PRIMARY_KEY_WEIGHT + REMAINING_FIELDS_WEIGHT == 1.0

    def test_primary_key_weight_is_positive(self):
        """Primary key weight should be positive."""
        assert PRIMARY_KEY_WEIGHT > 0

    def test_remaining_fields_weight_is_positive(self):
        """Remaining fields weight should be positive."""
        assert REMAINING_FIELDS_WEIGHT > 0


class TestFieldDefinitions:
    """Tests for field definitions."""

    def test_agreement_fields_not_empty(self):
        """AGREEMENT_FIELDS should not be empty."""
        assert len(AGREEMENT_FIELDS) > 0

    def test_agreement_fields_contains_expected(self):
        """AGREEMENT_FIELDS should contain expected fields."""
        expected = ["direction", "position_status", "exposure_change", "state_type"]
        for field in expected:
            assert field in AGREEMENT_FIELDS

    def test_field_values_has_direction(self):
        """FIELD_VALUES should have direction field."""
        assert "direction" in FIELD_VALUES
        assert "Long" in FIELD_VALUES["direction"]
        assert "Short" in FIELD_VALUES["direction"]
        assert "Unclear" in FIELD_VALUES["direction"]

    def test_field_values_has_exposure_change(self):
        """FIELD_VALUES should have exposure_change field."""
        assert "exposure_change" in FIELD_VALUES
        assert "Increase" in FIELD_VALUES["exposure_change"]
        assert "Decrease" in FIELD_VALUES["exposure_change"]
        assert "No Change" in FIELD_VALUES["exposure_change"]

    def test_field_values_has_state_type(self):
        """FIELD_VALUES should have state_type field."""
        assert "state_type" in FIELD_VALUES
        assert "Explicit State" in FIELD_VALUES["state_type"]
        assert "Direct State" in FIELD_VALUES["state_type"]
        assert "Indirect State" in FIELD_VALUES["state_type"]


class TestAmbiguousLabels:
    """Tests for ambiguous labels configuration."""

    def test_unclear_is_ambiguous(self):
        """'Unclear' should be marked as an ambiguous label."""
        assert "Unclear" in AMBIGUOUS_LABELS

    def test_unclear_appears_in_multiple_fields(self):
        """'Unclear' should map to multiple fields."""
        assert len(AMBIGUOUS_LABELS["Unclear"]) >= 2
        assert "direction" in AMBIGUOUS_LABELS["Unclear"]
        assert "exposure_change" in AMBIGUOUS_LABELS["Unclear"]

    def test_label_columns_has_disambiguated_unclear(self):
        """LABEL_COLUMNS should have field-specific Unclear columns."""
        assert "Unclear (direction)" in LABEL_COLUMNS
        assert "Unclear (exposure_change)" in LABEL_COLUMNS
        assert "Unclear (remaining_exposure)" in LABEL_COLUMNS

    def test_label_columns_no_bare_unclear(self):
        """LABEL_COLUMNS should not have bare 'Unclear' (should be disambiguated)."""
        # Bare "Unclear" should not be in LABEL_COLUMNS
        unclear_count = LABEL_COLUMNS.count("Unclear")
        assert unclear_count == 0, "Bare 'Unclear' should not appear in LABEL_COLUMNS"


class TestFieldColumns:
    """Tests for field column definitions."""

    def test_field_columns_match_agreement_fields(self):
        """FIELD_COLUMNS should match AGREEMENT_FIELDS."""
        for col in FIELD_COLUMNS:
            assert col in AGREEMENT_FIELDS


class TestValidationRules:
    """Tests for validation rules."""

    def test_direction_validation(self):
        """Direction validation should include expected values."""
        assert "Long" in VALIDATION_RULES["direction"]
        assert "Short" in VALIDATION_RULES["direction"]
        assert "Unclear" in VALIDATION_RULES["direction"]

    def test_label_type_validation(self):
        """Label type validation should include state and action."""
        assert "state" in VALIDATION_RULES["label_type"]
        assert "action" in VALIDATION_RULES["label_type"]

    def test_state_type_validation(self):
        """State type validation should include expected values."""
        assert "Explicit State" in VALIDATION_RULES["state_type"]
        assert "Direct State" in VALIDATION_RULES["state_type"]
        assert "Indirect State" in VALIDATION_RULES["state_type"]
