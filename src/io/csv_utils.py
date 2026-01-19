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


def add_total_rows(df: pl.DataFrame, is_gt_counts: bool = False) -> pl.DataFrame:
    """Add 'Total' rows aggregating across all traders using weighted mean.

    For each (primary_annotator, secondary_annotator) pair, creates a row with
    trader='Total' containing the weighted mean of scores across all traders
    (excluding rows where common_tasks == 0).

    Weighted mean = sum(score * common_tasks) / sum(common_tasks)
    This gives more weight to traders with more tasks, matching the pipeline's
    natural aggregation behavior.

    prim_annot_tasks and common_tasks are summed across traders for Total rows.
    """
    if "primary_annotator" not in df.columns or "secondary_annotator" not in df.columns:
        return df

    if "trader" not in df.columns:
        return df

    # Filter out rows with no common tasks (these have 0.0 scores that shouldn't count)
    # and exclude any existing Total rows
    df_with_data = df.filter(
        (pl.col("common_tasks") > 0) & (pl.col("trader") != "Total")
    )

    if df_with_data.height == 0:
        return df

    # Get numeric columns to aggregate
    numeric_cols = [col for col in df.columns if col not in STRING_COLUMNS]

    if is_gt_counts:
        # For gt_counts, sum all numeric columns
        agg_exprs = [pl.col(col).sum() for col in numeric_cols]
    else:
        # For agreement scores, use weighted mean (weighted by common_tasks)
        # Sum task count columns (prim_annot_tasks, common_tasks)
        # Weighted mean for score columns: sum(score * tasks) / sum(tasks)
        sum_cols = [col for col in numeric_cols if col in SUM_COLUMNS]
        mean_cols = [col for col in numeric_cols if col not in SUM_COLUMNS]
        agg_exprs = [pl.col(col).sum() for col in sum_cols]
        # Create weighted sum expressions for score columns
        agg_exprs += [
            (pl.col(col) * pl.col("common_tasks")).sum().alias(f"_{col}_weighted")
            for col in mean_cols
        ]

    # Aggregate per annotator pair across all traders
    total_rows = df_with_data.group_by(
        ["primary_annotator", "secondary_annotator"], maintain_order=True
    ).agg(agg_exprs)

    # For non-gt_counts, compute final weighted means by dividing by total common_tasks
    if not is_gt_counts:
        mean_cols = [col for col in numeric_cols if col not in SUM_COLUMNS]
        total_rows = total_rows.with_columns(
            [
                (pl.col(f"_{col}_weighted") / pl.col("common_tasks")).alias(col)
                for col in mean_cols
            ]
        ).drop([f"_{col}_weighted" for col in mean_cols])

    # Add trader='Total'
    total_rows = total_rows.with_columns(pl.lit("Total").alias("trader"))

    # Reorder columns to match original dataframe
    total_rows = total_rows.select(df.columns)

    # Remove any existing Total rows from original df and add new ones
    df_without_total = df.filter(pl.col("trader") != "Total")

    return pl.concat([df_without_total, total_rows], how="vertical")
