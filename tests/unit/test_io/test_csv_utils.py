"""Tests for src/io/csv_utils.py."""

import polars as pl
import pytest

from src.io.csv_utils import (
    PRIORITY_COLUMNS,
    STRING_COLUMNS,
    SUM_COLUMNS,
    add_per_trader_rows,
    reorder_columns,
)


class TestConstants:
    """Tests for CSV utility constants."""

    def test_string_columns_defined(self):
        """STRING_COLUMNS should contain expected column names."""
        assert "annotator" in STRING_COLUMNS
        assert "trader" in STRING_COLUMNS
        assert "primary_annotator" in STRING_COLUMNS
        assert "secondary_annotator" in STRING_COLUMNS

    def test_sum_columns_defined(self):
        """SUM_COLUMNS should contain expected column names."""
        assert "prim_annot_tasks" in SUM_COLUMNS
        assert "common_tasks" in SUM_COLUMNS
        assert "num_tasks" in SUM_COLUMNS

    def test_priority_columns_defined(self):
        """PRIORITY_COLUMNS should contain expected column names."""
        assert "trader" in PRIORITY_COLUMNS
        assert "primary_annotator" in PRIORITY_COLUMNS
        assert "secondary_annotator" in PRIORITY_COLUMNS


class TestReorderColumns:
    """Tests for reorder_columns function."""

    def test_reorders_with_priority_first(self):
        """Should place priority columns first."""
        df = pl.DataFrame(
            {
                "direction": [0.8],
                "trader": ["A"],
                "exposure_change": [0.7],
                "primary_annotator": ["user1"],
            }
        )
        result = reorder_columns(df)

        columns = result.columns
        # trader and primary_annotator should be before direction
        assert columns.index("trader") < columns.index("direction")
        assert columns.index("primary_annotator") < columns.index("direction")

    def test_preserves_all_columns(self):
        """Should preserve all original columns."""
        df = pl.DataFrame(
            {
                "direction": [0.8],
                "trader": ["A"],
                "exposure_change": [0.7],
                "custom_col": [1],
            }
        )
        result = reorder_columns(df)

        assert set(result.columns) == set(df.columns)

    def test_handles_missing_priority_columns(self):
        """Should handle DataFrames without all priority columns."""
        df = pl.DataFrame({"direction": [0.8], "trader": ["A"]})
        result = reorder_columns(df)

        assert result.columns[0] == "trader"
        assert "direction" in result.columns

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrames."""
        df = pl.DataFrame({"trader": [], "direction": []}).cast(
            {"trader": pl.String, "direction": pl.Float64}
        )
        result = reorder_columns(df)

        assert result.shape[0] == 0
        assert "trader" in result.columns


class TestAddPerTraderRows:
    """Tests for add_per_trader_rows function."""

    def test_adds_aggregated_rows(self, sample_agreement_df):
        """Should add aggregated rows per trader."""
        result = add_per_trader_rows(sample_agreement_df)

        # Original 4 rows + 2 aggregated rows (one per trader)
        original_count = sample_agreement_df.shape[0]
        assert result.shape[0] > original_count

        # Check that aggregated rows exist
        agg_rows = result.filter(pl.col("primary_annotator") == "ALL")
        assert agg_rows.shape[0] == 2  # Two traders

    def test_aggregated_rows_have_null_secondary(self, sample_agreement_df):
        """Aggregated rows should have null secondary_annotator."""
        result = add_per_trader_rows(sample_agreement_df)

        agg_rows = result.filter(pl.col("primary_annotator") == "ALL")
        assert all(val is None for val in agg_rows["secondary_annotator"].to_list())

    def test_avoids_double_counting(self, sample_agreement_df):
        """Should deduplicate pairs before aggregation."""
        result = add_per_trader_rows(sample_agreement_df)

        # Check aggregated values are reasonable
        # user1-user2 and user2-user1 should only count once
        agg_trader_a = result.filter(
            (pl.col("primary_annotator") == "ALL") & (pl.col("trader") == "trader_A")
        )
        # Should be average of one pair, not two
        assert agg_trader_a.shape[0] == 1

    def test_sums_task_columns(self, sample_agreement_df):
        """Should sum prim_annot_tasks for aggregation."""
        result = add_per_trader_rows(sample_agreement_df)

        # For trader_A, the deduped pair should have sum of one annotator's tasks
        agg_trader_a = result.filter(
            (pl.col("primary_annotator") == "ALL") & (pl.col("trader") == "trader_A")
        )
        # After deduplication, we keep one of (user1-user2, user2-user1)
        # prim_annot_tasks for trader_A: user1=15, user2=12
        # Only one is kept, so sum equals that one value
        prim_tasks = agg_trader_a["prim_annot_tasks"][0]
        assert prim_tasks in [15, 12]  # One of the original values

    def test_averages_score_columns(self, sample_agreement_df):
        """Should average score columns for aggregation."""
        result = add_per_trader_rows(sample_agreement_df)

        agg_trader_a = result.filter(
            (pl.col("primary_annotator") == "ALL") & (pl.col("trader") == "trader_A")
        )
        # After dedup, only one row per trader, so mean = that value
        direction_score = agg_trader_a["direction"][0]
        assert direction_score == 0.8  # Same as original pairs

    def test_gt_counts_sums_all_numeric(self, sample_gt_counts_df):
        """With is_gt_counts=True, should sum all numeric columns."""
        result = add_per_trader_rows(sample_gt_counts_df, is_gt_counts=True)

        # Only one trader in sample, so one aggregated row
        agg_row = result.filter(pl.col("primary_annotator") == "ALL")
        assert agg_row.shape[0] == 1

        # Long column should be summed (but after dedup, only one row)
        # user1-gt and user2-gt are different pairs, both kept
        long_sum = agg_row["Long"][0]
        # After dedup by canonical pair, both rows kept (different secondary)
        assert long_sum == 15 + 10  # Both rows summed

    def test_returns_original_if_no_annotator_columns(self):
        """Should return original DataFrame if no annotator columns."""
        df = pl.DataFrame({"trader": ["A"], "score": [0.8]})
        result = add_per_trader_rows(df)

        assert result.shape == df.shape

    def test_preserves_original_rows(self, sample_agreement_df):
        """Should preserve all original rows."""
        result = add_per_trader_rows(sample_agreement_df)

        # Original rows should still be present
        original_rows = result.filter(pl.col("primary_annotator") != "ALL")
        assert original_rows.shape[0] == sample_agreement_df.shape[0]
