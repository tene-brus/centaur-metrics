import argparse
import os

import polars as pl


STRING_COLUMNS = ["annotator", "trader", "primary_annotator", "secondary_annotator"]


def merge_csvs_in_directory(directory: str) -> None:
    """Merge all CSVs in a directory and save to parent directory."""
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

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

    merged = pl.concat(dfs, how="vertical")

    # Normalize path to handle trailing slashes
    directory = os.path.normpath(directory)
    parent_dir = os.path.dirname(directory)
    subdir_name = os.path.basename(directory)
    output_file = os.path.join(parent_dir, f"merged_{subdir_name}.csv")

    print(output_file)
    merged.write_csv(output_file, float_precision=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all CSVs in a directory")
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing CSV files to merge",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    merge_csvs_in_directory(args.directory)


if __name__ == "__main__":
    main()
