import argparse
import os

import polars as pl

STRING_COLUMNS = ["annotator", "trader", "primary_annotator", "secondary_annotator"]
SUM_COLUMNS = ["prim_annot_tasks", "common_tasks", "num_tasks"]


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
    file1: str, file2: str, is_overall_agreement: bool = False
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


def combine_projects(dir1: str, dir2: str, output_dir: str) -> None:
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

        # Check if this is an overall agreement file
        is_overall_agreement = "overall_agreement" in key

        combined = add_stats(file1, file2, is_overall_agreement)

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
    flatten_combined_csvs(output_dir)


def flatten_combined_csvs(output_dir: str) -> None:
    """Create flattened CSV files with subdirectory names included in filename."""
    flat_dir = os.path.join(output_dir, "flat")
    os.makedirs(flat_dir, exist_ok=True)

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

                # Copy the file
                df = pl.read_csv(src_path)
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

    args = parser.parse_args()

    if not os.path.isdir(args.dir1):
        print(f"Error: {args.dir1} is not a valid directory")
        return

    if not os.path.isdir(args.dir2):
        print(f"Error: {args.dir2} is not a valid directory")
        return

    combine_projects(args.dir1, args.dir2, args.output_dir)


if __name__ == "__main__":
    main()
