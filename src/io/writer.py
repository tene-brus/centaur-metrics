"""CSV writing utilities."""

import polars as pl

from src.io.paths import CaseType, OutputConfig
from src.models.constants import FIELD_COLUMNS, LABEL_COLUMNS


class CSVWriter:
    """
    Writes metrics results to CSV files.

    Handles formatting and schema creation for different result types.
    """

    def __init__(self, config: OutputConfig, float_precision: int = 3):
        """
        Initialize writer with output configuration.

        Args:
            config: Output path configuration
            float_precision: Number of decimal places for floats
        """
        self.config = config
        self.float_precision = float_precision

    def write_overall(
        self,
        scores: dict[str, dict[str, float | None]],
        data: pl.DataFrame,
        trader: str | None = None,
    ) -> str:
        """
        Write overall agreement results to CSV.

        Args:
            scores: Dict of annotator -> {annotator2 -> score}
            data: Original DataFrame for task counts
            trader: Trader name (None for total)

        Returns:
            Path to written file
        """
        self.config.ensure_dirs()

        # Build DataFrame from scores
        df = pl.DataFrame(schema={key: pl.Float64 for key in scores.keys()})
        for annotator in scores.keys():
            df = df.extend(pl.from_dict(scores[annotator]))

        # Drop ground_truth column and add mean
        if "ground_truth" in df.columns:
            df = df.drop("ground_truth")

        df = df.with_columns(pl.mean_horizontal(pl.all()).alias("mean_agreement"))

        # Add annotator column
        col_df = pl.DataFrame({"annotator": df.columns[:-1]})
        final = pl.concat([col_df, df], how="horizontal")

        # Add task counts
        annotator_tasks = []
        for col in final.columns:
            if col in ["ground_truth_member", "num_annotations", "annotator", "mean_agreement"]:
                if col in ["ground_truth_member", "num_annotations"]:
                    annotator_tasks.append(0)
                continue
            else:
                num_tasks = data.filter(pl.col(col).is_not_null())
                if trader:
                    num_tasks = num_tasks.filter(pl.col("trader") == trader)
                annotator_tasks.append(num_tasks.shape[0])

        tasks_df = pl.DataFrame({"num_tasks": annotator_tasks})
        final = pl.concat([final, tasks_df], how="horizontal")

        # Add trader column
        trader_value = trader if trader is not None else "Total"
        final = final.with_columns(pl.lit(trader_value).alias("trader"))

        # Write
        output_path = self.config.get_output_path(trader)
        final.filter(pl.col("annotator").is_not_null()).write_csv(
            output_path, float_precision=self.float_precision
        )

        print(output_path)
        return output_path

    def write_per_field_or_label(
        self,
        results: dict[str, dict[str, dict[str, float]]],
        data: pl.DataFrame,
        trader: str | None = None,
        counts: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> str:
        """
        Write per-field or per-label agreement results to CSV.

        Args:
            results: Dict of annotator -> {annotator2 -> {field/label -> score}}
            data: Original DataFrame for task counts
            trader: Trader name (None for total)
            counts: Optional counts dict for per-label case

        Returns:
            Path to written file
        """
        self.config.ensure_dirs()

        columns = FIELD_COLUMNS if self.config.case == CaseType.FIELD else LABEL_COLUMNS
        tables: dict[str, pl.DataFrame] = {}

        for annotator in results.keys():
            prim_annot_tasks = data.filter(pl.col(annotator).is_not_null()).shape[0]

            for secondary, inner_dict in results[annotator].items():
                # Calculate common tasks
                if secondary == "ground_truth" and not self.config.common:
                    common_tasks = (
                        data.filter(pl.col(annotator).is_not_null())
                        .filter(pl.col(secondary).is_not_null())
                        .filter(pl.col("ground_truth_member") != annotator)
                        .shape[0]
                    )
                else:
                    common_tasks = (
                        data.filter(pl.col(annotator).is_not_null())
                        .filter(pl.col(secondary).is_not_null())
                        .shape[0]
                    )

                # Ensure all columns exist
                if len(inner_dict) < 3:
                    inner_dict = {key: 0.0 for key in columns}

                row_data = dict(inner_dict)
                row_data["primary_annotator"] = annotator
                row_data["secondary_annotator"] = secondary
                row_data["prim_annot_tasks"] = prim_annot_tasks
                row_data["common_tasks"] = common_tasks

                # Create or extend table
                if annotator not in tables:
                    schema = {
                        key: pl.Float64
                        for key in row_data.keys()
                        if key not in ["primary_annotator", "secondary_annotator", "prim_annot_tasks", "common_tasks"]
                    }
                    schema["primary_annotator"] = pl.String
                    schema["secondary_annotator"] = pl.String
                    schema["prim_annot_tasks"] = pl.Int64
                    schema["common_tasks"] = pl.Int64
                    tables[annotator] = pl.DataFrame(schema=schema)

                tables[annotator].extend(
                    pl.from_dict(row_data).select(tables[annotator].columns)
                )

        # Combine all annotator tables
        schema = {key: pl.Float64 for key in columns}
        schema["primary_annotator"] = pl.String
        schema["secondary_annotator"] = pl.String
        schema["prim_annot_tasks"] = pl.Int64
        schema["common_tasks"] = pl.Int64
        master_table = pl.DataFrame(schema=schema)

        for table in tables.values():
            master_table.extend(table.select(master_table.columns))

        # Add trader column
        trader_value = trader if trader is not None else "Total"
        master_table = master_table.with_columns(pl.lit(trader_value).alias("trader"))

        # Write main file
        output_path = self.config.get_output_path(trader)
        master_table.write_csv(output_path, float_precision=self.float_precision)
        print(output_path)

        # Write supplementary files
        if self.config.case == CaseType.FIELD:
            self._write_gt_breakdown(master_table, trader)
        elif self.config.case == CaseType.LABEL and counts is not None:
            self._write_gt_counts(counts, data, trader)

        return output_path

    def _write_gt_breakdown(self, df: pl.DataFrame, trader: str | None) -> str:
        """Write ground truth breakdown for field case."""
        gt_breakdown = (
            df.filter(pl.col("secondary_annotator") == "ground_truth")
            .with_columns(pl.col(FIELD_COLUMNS) * 5)
            .with_columns(
                pl.mean_horizontal(
                    pl.all().exclude([
                        "primary_annotator",
                        "secondary_annotator",
                        "prim_annot_tasks",
                        "common_tasks",
                        "trader",
                    ])
                ).alias("sum_contrib")
            )
        )

        output_path = self.config.get_gt_breakdown_path(trader)
        gt_breakdown.write_csv(output_path, float_precision=self.float_precision)
        print(output_path)
        return output_path

    def _write_gt_counts(
        self,
        counts: dict[str, dict[str, dict[str, float]]],
        data: pl.DataFrame,
        trader: str | None,
    ) -> str:
        """Write ground truth counts for label case."""
        # Build counts DataFrame similar to results
        columns = LABEL_COLUMNS
        tables: dict[str, pl.DataFrame] = {}

        for annotator in counts.keys():
            prim_annot_tasks = data.filter(pl.col(annotator).is_not_null()).shape[0]

            for secondary, inner_dict in counts[annotator].items():
                if secondary == "ground_truth" and not self.config.common:
                    common_tasks = (
                        data.filter(pl.col(annotator).is_not_null())
                        .filter(pl.col(secondary).is_not_null())
                        .filter(pl.col("ground_truth_member") != annotator)
                        .shape[0]
                    )
                else:
                    common_tasks = (
                        data.filter(pl.col(annotator).is_not_null())
                        .filter(pl.col(secondary).is_not_null())
                        .shape[0]
                    )

                if len(inner_dict) < 3:
                    inner_dict = {key: 0.0 for key in columns}

                row_data = dict(inner_dict)
                row_data["primary_annotator"] = annotator
                row_data["secondary_annotator"] = secondary
                row_data["prim_annot_tasks"] = prim_annot_tasks
                row_data["common_tasks"] = common_tasks

                if annotator not in tables:
                    schema = {
                        key: pl.Float64
                        for key in row_data.keys()
                        if key not in ["primary_annotator", "secondary_annotator", "prim_annot_tasks", "common_tasks"]
                    }
                    schema["primary_annotator"] = pl.String
                    schema["secondary_annotator"] = pl.String
                    schema["prim_annot_tasks"] = pl.Int64
                    schema["common_tasks"] = pl.Int64
                    tables[annotator] = pl.DataFrame(schema=schema)

                tables[annotator].extend(
                    pl.from_dict(row_data).select(tables[annotator].columns)
                )

        schema = {key: pl.Float64 for key in columns}
        schema["primary_annotator"] = pl.String
        schema["secondary_annotator"] = pl.String
        schema["prim_annot_tasks"] = pl.Int64
        schema["common_tasks"] = pl.Int64
        master_table = pl.DataFrame(schema=schema)

        for table in tables.values():
            master_table.extend(table.select(master_table.columns))

        trader_value = trader if trader is not None else "Total"
        master_table = master_table.with_columns(pl.lit(trader_value).alias("trader"))

        # Filter to ground truth only
        gt_counts = master_table.filter(pl.col("secondary_annotator") == "ground_truth")

        output_path = self.config.get_gt_counts_path(trader)
        gt_counts.write_csv(output_path)
        print(output_path)
        return output_path
