import argparse
import os

import polars as pl

from src.io.csv_utils import (
    STRING_COLUMNS,
    add_per_trader_rows,
    add_total_rows,
    reorder_columns,
)


def get_trader_task_counts(jsonl_path: str) -> pl.DataFrame:
    """Calculate total tasks per trader from source JSONL file.

    Also includes a 'Total' row with the total count across all traders.
    """
    if not os.path.exists(jsonl_path):
        return pl.DataFrame({"trader": [], "total_tasks": []})

    data = pl.read_ndjson(jsonl_path, infer_schema_length=8000)
    if "trader" not in data.columns:
        return pl.DataFrame({"trader": [], "total_tasks": []})

    per_trader = data.group_by("trader").agg(
        pl.len().cast(pl.Int64).alias("total_tasks")
    )

    # Add a "Total" row with the sum of all tasks
    total_row = pl.DataFrame({"trader": ["Total"], "total_tasks": [len(data)]})

    return pl.concat([per_trader, total_row], how="vertical")


def merge_csvs_in_directory(directory: str, jsonl_path: str | None = None) -> None:
    """Merge all CSVs in a directory and save to parent directory."""
    # Exclude Total_agreement.csv and any existing merged_*.csv files
    csv_files = [
        f
        for f in os.listdir(directory)
        if f.endswith(".csv")
        and f != "Total_agreement.csv"
        and not f.startswith("merged_")
    ]

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        df = pl.read_csv(file_path)

        # Cast all-null string columns to Float64 (except known string columns)
        for col in df.columns:
            if col not in STRING_COLUMNS and df[col].dtype == pl.String:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        dfs.append(df)

    # Use diagonal concat to handle CSVs with different column sets
    merged = pl.concat(dfs, how="diagonal")

    # Add per-trader aggregation and Total rows for per_label and per_field directories
    is_per_label_or_field = "per_label" in directory or "per_field" in directory
    is_gt_counts = "gt_counts" in directory
    if is_per_label_or_field:
        merged = add_per_trader_rows(merged, is_gt_counts=is_gt_counts)
        # Add Total rows using simple mean (not weighted by task count)
        merged = add_total_rows(merged, is_gt_counts=is_gt_counts)

        # If JSONL path provided, fix task counts from source data
        if jsonl_path and "prim_annot_tasks" in merged.columns:
            task_counts = get_trader_task_counts(jsonl_path)

            # Update ALL rows (aggregated per trader) with correct task counts
            all_rows = merged.filter(pl.col("primary_annotator") == "ALL")
            other_rows = merged.filter(pl.col("primary_annotator") != "ALL")

            # Join with task counts for ALL rows
            all_rows = (
                all_rows.join(
                    task_counts.rename({"total_tasks": "_jsonl_tasks"}),
                    on="trader",
                    how="left",
                )
                .with_columns(
                    pl.coalesce(
                        pl.col("_jsonl_tasks"), pl.col("prim_annot_tasks")
                    ).alias("prim_annot_tasks")
                )
                .drop("_jsonl_tasks")
                .select(merged.columns)
            )

            merged = pl.concat([other_rows, all_rows], how="vertical")

    # Reorder columns with priority columns first
    merged = reorder_columns(merged)

    # Normalize path to handle trailing slashes
    directory = os.path.normpath(directory)
    subdir_name = os.path.basename(directory)
    output_file = os.path.join(directory, f"merged_{subdir_name}.csv")

    print(output_file)
    merged.write_csv(output_file, float_precision=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all CSVs in a directory")
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing CSV files to merge",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=None,
        help="Path to source JSONL file for calculating task counts",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    merge_csvs_in_directory(args.directory, args.jsonl_path)


if __name__ == "__main__":
    main()
