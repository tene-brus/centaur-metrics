"""CSV utilities for column management and aggregation."""

import polars as pl

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

STRING_COLUMNS = ["annotator", "trader", "primary_annotator", "secondary_annotator"]
SUM_COLUMNS = ["prim_annot_tasks", "common_tasks", "num_tasks"]
PRIORITY_COLUMNS = [
    "trader",
    "primary_annotator",
    "secondary_annotator",
    "prim_annot_tasks",
    "common_tasks",
]


# ============================================================================
# COLUMN UTILITIES
# ============================================================================


def reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder columns with priority columns first, then the rest."""
    priority = [col for col in PRIORITY_COLUMNS if col in df.columns]
    remaining = [col for col in df.columns if col not in PRIORITY_COLUMNS]
    return df.select(priority + remaining)


# ============================================================================
# AGGREGATION UTILITIES
# ============================================================================


def add_per_trader_rows(df: pl.DataFrame, is_gt_counts: bool = False) -> pl.DataFrame:
    """Add per-trader aggregated rows to dataframes with primary/secondary annotator pairs.

    Since each annotator pair (A, B) appears twice (as A-B and B-A), we create a canonical
    pair by sorting alphabetically and only keep one instance before aggregating.
    """
    if "primary_annotator" not in df.columns or "secondary_annotator" not in df.columns:
        return df

    # Create canonical annotator pair (sorted alphabetically) to avoid double counting
    df_with_canonical = df.with_columns(
        [
            pl.when(pl.col("primary_annotator") < pl.col("secondary_annotator"))
            .then(pl.col("primary_annotator"))
            .otherwise(pl.col("secondary_annotator"))
            .alias("_annot_1"),
            pl.when(pl.col("primary_annotator") < pl.col("secondary_annotator"))
            .then(pl.col("secondary_annotator"))
            .otherwise(pl.col("primary_annotator"))
            .alias("_annot_2"),
        ]
    )

    # Deduplicate by keeping only one row per (trader, canonical_pair)
    df_deduped = df_with_canonical.unique(
        subset=["trader", "_annot_1", "_annot_2"], keep="first"
    )

    # Get numeric columns to aggregate
    numeric_cols = [col for col in df.columns if col not in STRING_COLUMNS]

    if is_gt_counts:
        # For gt_counts, sum all numeric columns
        agg_exprs = [pl.col(col).sum() for col in numeric_cols]
    else:
        # For agreement scores, use mean aggregation
        # Sum task count columns, mean for score columns
        sum_cols = [col for col in numeric_cols if col in SUM_COLUMNS]
        mean_cols = [col for col in numeric_cols if col not in SUM_COLUMNS]
        agg_exprs = [pl.col(col).sum() for col in sum_cols]
        agg_exprs += [pl.col(col).mean() for col in mean_cols]

    # Aggregate per trader
    per_trader = df_deduped.group_by("trader", maintain_order=True).agg(agg_exprs)

    # Add placeholder columns for annotators and set common_tasks to null
    per_trader = per_trader.with_columns(
        [
            pl.lit("ALL").alias("primary_annotator"),
            pl.lit(None).alias("secondary_annotator"),
        ]
    )
    if "common_tasks" in per_trader.columns:
        per_trader = per_trader.with_columns(pl.lit(None).alias("common_tasks"))

    # Reorder columns to match original dataframe
    per_trader = per_trader.select(df.columns)

    # Concatenate original dataframe with per-trader rows
    return pl.concat([df, per_trader], how="vertical")
