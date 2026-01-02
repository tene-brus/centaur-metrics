import argparse
import os

import polars as pl

from src.io.csv_utils import (
    STRING_COLUMNS,
    SUM_COLUMNS,
    add_per_trader_rows,
    add_total_rows,
    reorder_columns,
)


def find_merged_csvs(directory: str) -> dict[str, str]:
    """Find all merged CSV files in a directory and its subdirectories."""
    merged_files = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("merged_") and file.endswith(".csv"):
                # Get relative path from the base directory
                rel_path = os.path.relpath(root, directory)
                key = os.path.join(rel_path, file)
                merged_files[key] = os.path.join(root, file)

    return merged_files


def add_stats(
    file1: str,
    file2: str,
    is_overall_agreement: bool = False,
    is_gt_counts: bool = False,
) -> pl.DataFrame:
    """Add numeric stats from two CSVs, grouping by trader (and annotator columns)."""
    df1 = pl.read_csv(file1)
    df2 = pl.read_csv(file2)

    # Cast all non-string columns to Float64 before concatenation
    for col in df1.columns:
        if col not in STRING_COLUMNS:
            df1 = df1.with_columns(pl.col(col).cast(pl.Float64))
    for col in df2.columns:
        if col not in STRING_COLUMNS:
            df2 = df2.with_columns(pl.col(col).cast(pl.Float64))

    # For overall agreement files, just concatenate vertically (different annotator columns)
    if is_overall_agreement:
        combined = pl.concat([df1, df2], how="diagonal")
    else:
        # Concatenate and group by string columns, summing numeric columns
        combined = pl.concat([df1, df2], how="vertical")

    # Cast all non-string columns to Float64 after concatenation (for diagonal concat)
    for col in combined.columns:
        if col not in STRING_COLUMNS:
            combined = combined.with_columns(pl.col(col).cast(pl.Float64))

    # Determine group by columns (string columns that exist in the dataframe)
    group_cols = [col for col in STRING_COLUMNS if col in combined.columns]

    if is_gt_counts:
        # For gt_counts files, sum all numeric columns (they are counts)
        sum_cols = [col for col in combined.columns if col not in STRING_COLUMNS]
        agg_exprs = [pl.col(col).sum() for col in sum_cols]
    else:
        # Determine columns to sum vs mean
        sum_cols = [col for col in combined.columns if col in SUM_COLUMNS]
        mean_cols = [
            col
            for col in combined.columns
            if col not in STRING_COLUMNS and col not in SUM_COLUMNS
        ]

        # Aggregate: sum task columns, mean for other numeric columns
        agg_exprs = [pl.col(col).sum() for col in sum_cols]
        agg_exprs += [pl.col(col).mean() for col in mean_cols]

    aggregated = combined.group_by(group_cols, maintain_order=True).agg(agg_exprs)

    return aggregated


def get_combined_trader_task_counts(jsonl_paths: list[str]) -> pl.DataFrame:
    """Calculate total tasks per trader from multiple JSONL files.

    Also includes a 'Total' row with the total count across all traders and files.
    """
    all_traders = []
    total_count = 0

    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            continue
        data = pl.read_ndjson(jsonl_path, infer_schema_length=8000)
        if "trader" not in data.columns:
            continue
        all_traders.append(data.select(["trader"]))
        total_count += len(data)

    if not all_traders:
        return pl.DataFrame({"trader": [], "total_tasks": []})

    combined = pl.concat(all_traders, how="vertical")
    per_trader = combined.group_by("trader").agg(pl.len().cast(pl.Int64).alias("total_tasks"))

    # Add a "Total" row with the sum of all tasks
    total_row = pl.DataFrame({"trader": ["Total"], "total_tasks": [total_count]})

    return pl.concat([per_trader, total_row], how="vertical")


def combine_projects(
    dir1: str, dir2: str, output_dir: str, jsonl_paths: list[str] | None = None
) -> None:
    """Combine matching merged CSVs from two project directories by adding stats."""
    dir1 = os.path.normpath(dir1)
    dir2 = os.path.normpath(dir2)

    merged1 = find_merged_csvs(dir1)
    merged2 = find_merged_csvs(dir2)

    # Find common keys (matching subdirectory structure)
    common_keys = set(merged1.keys()) & set(merged2.keys())

    if not common_keys:
        print("No matching merged CSV files found between directories")
        return

    os.makedirs(output_dir, exist_ok=True)

    for key in sorted(common_keys):
        file1 = merged1[key]
        file2 = merged2[key]

        # Check if this is an overall agreement file or gt_counts file
        is_overall_agreement = "overall_agreement" in key
        is_gt_counts = "gt_counts" in key

        combined = add_stats(file1, file2, is_overall_agreement, is_gt_counts)

        # Reorder columns with priority columns first
        combined = reorder_columns(combined)

        # Create output path preserving subdirectory structure
        output_path = os.path.join(output_dir, key)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(output_path)
        combined.write_csv(output_path, float_precision=3)

    # Report any files that didn't have a match
    only_in_1 = set(merged1.keys()) - set(merged2.keys())
    only_in_2 = set(merged2.keys()) - set(merged1.keys())

    if only_in_1:
        print(f"\nFiles only in {dir1}:")
        for key in sorted(only_in_1):
            print(f"  {key}")

    if only_in_2:
        print(f"\nFiles only in {dir2}:")
        for key in sorted(only_in_2):
            print(f"  {key}")

    # Create flattened CSVs with subdirectory names in filename
    flatten_combined_csvs(output_dir, jsonl_paths)


def flatten_combined_csvs(output_dir: str, jsonl_paths: list[str] | None = None) -> None:
    """Create flattened CSV files with subdirectory names included in filename."""
    flat_dir = os.path.join(output_dir, "flat")
    os.makedirs(flat_dir, exist_ok=True)

    # Get task counts from JSONL files if provided
    task_counts = None
    if jsonl_paths:
        task_counts = get_combined_trader_task_counts(jsonl_paths)

    for root, _, files in os.walk(output_dir):
        # Skip the flat directory itself
        if "flat" in root:
            continue

        for file in files:
            if file.endswith(".csv"):
                # Get relative path from output_dir
                rel_path = os.path.relpath(root, output_dir)

                # Create flattened filename: subdir1_subdir2_filename.csv
                if rel_path != ".":
                    subdir_parts = rel_path.replace(os.sep, "_")
                    flat_filename = f"{subdir_parts}_{file}"
                else:
                    flat_filename = file

                src_path = os.path.join(root, file)
                dst_path = os.path.join(flat_dir, flat_filename)

                # Read the file
                df = pl.read_csv(src_path)

                # Add per-trader aggregation and Total rows for per_label and per_field files
                is_per_label_or_field = "per_label" in file or "per_field" in file
                is_gt_counts = "gt_counts" in file
                if is_per_label_or_field:
                    df = add_per_trader_rows(df, is_gt_counts=is_gt_counts)
                    # Add Total rows using simple mean (not weighted by task count)
                    df = add_total_rows(df, is_gt_counts=is_gt_counts)

                    # Fix task counts from JSONL if provided
                    if task_counts is not None and "prim_annot_tasks" in df.columns:
                        all_rows = df.filter(pl.col("primary_annotator") == "ALL")
                        other_rows = df.filter(pl.col("primary_annotator") != "ALL")

                        all_rows = all_rows.join(
                            task_counts.rename({"total_tasks": "_jsonl_tasks"}),
                            on="trader",
                            how="left",
                        ).with_columns(
                            pl.coalesce(
                                pl.col("_jsonl_tasks"), pl.col("prim_annot_tasks")
                            ).alias("prim_annot_tasks")
                        ).drop("_jsonl_tasks").select(df.columns)

                        df = pl.concat([other_rows, all_rows], how="vertical")

                # Reorder columns with priority columns first
                df = reorder_columns(df)

                print(dst_path)
                df.write_csv(dst_path, float_precision=3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine merged CSVs from two project metrics directories by adding stats"
    )
    parser.add_argument(
        "--dir1",
        type=str,
        required=True,
        help="First project metrics directory",
    )
    parser.add_argument(
        "--dir2",
        type=str,
        required=True,
        help="Second project metrics directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="combined_metrics",
        help="Output directory for combined CSVs",
    )
    parser.add_argument(
        "--jsonl_paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to source JSONL files for calculating task counts",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dir1):
        print(f"Error: {args.dir1} is not a valid directory")
        return

    if not os.path.isdir(args.dir2):
        print(f"Error: {args.dir2} is not a valid directory")
        return

    combine_projects(args.dir1, args.dir2, args.output_dir, args.jsonl_paths)


if __name__ == "__main__":
    main()
