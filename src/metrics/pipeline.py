"""Main metrics computation pipeline."""

import os
from typing import Literal

import polars as pl

from src.io.loader import DataLoader
from src.metrics.pairwise import PairwiseCalculator
from src.models.constants import FIELD_COLUMNS, LABEL_COLUMNS


class MetricsPipeline:
    """
    Main orchestrator for metrics computation.

    Coordinates data loading, pairwise calculation, and output writing.
    This replaces the main() function logic in the original metrics.py.
    """

    def __init__(
        self,
        data_path: str,
        case: Literal["field", "label"] | None = None,
        common: bool = False,
        output_dir: str | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            data_path: Path to the JSONL data file
            case: Agreement calculation type (None=overall, "field", "label")
            common: Only compare on commonly-annotated tasks
            output_dir: Output directory (defaults to {data_basename}_metrics)
        """
        self.loader = DataLoader(data_path)
        self.case = case
        self.common = common

        # Set output directory
        if output_dir is None:
            output_dir = f"{self.loader.base_name}_metrics"

        self.output_dir = output_dir
        self.calculator = PairwiseCalculator.create(case=case, common=common)

    def run(self, per_trader: bool = False) -> None:
        """
        Execute the metrics pipeline.

        Args:
            per_trader: If True, generate separate CSV for each trader.
                       If False, generate single Total_agreement.csv.
        """
        data = self.loader.load()
        annotators = self.loader.annotators

        if per_trader:
            self._run_per_trader(data, annotators)
        else:
            self._run_total(data, annotators)

    def _run_total(self, data: pl.DataFrame, annotators: list[str]) -> None:
        """Run metrics for all data combined."""
        scores = self.calculator.calculate_all_pairs(data, annotators)

        subdir = self._get_output_subdir()
        os.makedirs(subdir, exist_ok=True)
        filename = "Total_agreement.csv"
        output_file = os.path.join(subdir, filename)

        if self.case is None:
            # Overall agreement
            result = self._sum_up_metrics(scores, data, trader=None)
            result.filter(pl.col("annotator").is_not_null()).write_csv(
                output_file, float_precision=3
            )
        elif self.case == "label":
            aggregated = self.calculator.aggregate_per_label_scores(
                scores, annotators, average=True
            )
            aggregated_counts = self.calculator.aggregate_per_label_scores(
                scores, annotators, average=False
            )

            result = self._sum_up_per_label_metrics(aggregated, data, trader=None)
            result.write_csv(output_file, float_precision=3)

            counts_result = self._sum_up_per_label_metrics(
                aggregated_counts, data, trader=None
            )
            self._create_gt_counts(counts_result, filename)
        else:  # field
            aggregated = self.calculator.aggregate_per_label_scores(
                scores, annotators, average=True
            )
            result = self._sum_up_per_label_metrics(aggregated, data, trader=None)
            result.write_csv(output_file, float_precision=3)
            self._create_gt_breakdown(result, filename)

        print(output_file)

    def _run_per_trader(self, data: pl.DataFrame, annotators: list[str]) -> None:
        """Run metrics for each trader separately."""
        subdir = self._get_output_subdir()
        os.makedirs(subdir, exist_ok=True)

        traders = data["trader"].unique().to_list()

        for trader in traders:
            trader_data = data.filter(pl.col("trader") == trader)

            if trader_data.shape[0] == 0:
                continue

            scores = self.calculator.calculate_all_pairs(trader_data, annotators)

            filename = f"agreement_{trader}.csv"
            output_file = os.path.join(subdir, filename)

            if self.case is None:
                result = self._sum_up_metrics(scores, trader_data, trader=trader)
                result.filter(pl.col("annotator").is_not_null()).write_csv(
                    output_file, float_precision=3
                )
            elif self.case == "label":
                aggregated = self.calculator.aggregate_per_label_scores(
                    scores, annotators, average=True
                )
                aggregated_counts = self.calculator.aggregate_per_label_scores(
                    scores, annotators, average=False
                )

                result = self._sum_up_per_label_metrics(
                    aggregated, trader_data, trader=trader
                )
                result.write_csv(output_file, float_precision=3)

                counts_result = self._sum_up_per_label_metrics(
                    aggregated_counts, trader_data, trader=trader
                )
                self._create_gt_counts(counts_result, filename)
            else:  # field
                aggregated = self.calculator.aggregate_per_label_scores(
                    scores, annotators, average=True
                )
                result = self._sum_up_per_label_metrics(
                    aggregated, trader_data, trader=trader
                )
                result.write_csv(output_file, float_precision=3)
                self._create_gt_breakdown(result, filename)

            print(output_file)

    def _get_output_subdir(self) -> str:
        """Get the output subdirectory path based on case and common flags."""
        if self.case is None:
            case_subdir = "overall_agreement"
        else:
            case_subdir = f"agreement_per_{self.case}"
        return os.path.join(self.output_dir, case_subdir, f"common_{self.common}")

    def _sum_up_metrics(
        self,
        result: dict[str, dict[str, float | None]],
        data: pl.DataFrame,
        trader: str | None = None,
    ) -> pl.DataFrame:
        """Create summary DataFrame for overall agreement."""
        df = pl.DataFrame(schema={key: pl.Float64 for key in result.keys()})

        for annotator in result.keys():
            df = df.extend(pl.from_dict(result[annotator]))

        if "ground_truth" in df.columns:
            df = df.drop("ground_truth")

        df = df.with_columns(pl.mean_horizontal(pl.all()).alias("mean_agreement"))

        col_df = pl.DataFrame({"annotator": df.columns[:-1]})
        final = pl.concat([col_df, df], how="horizontal")

        annotator_tasks = []
        for col in final.columns:
            if col in ["ground_truth_member", "num_annotations"]:
                annotator_tasks.append(0)
            elif col in ["annotator", "mean_agreement"]:
                continue
            else:
                num_tasks = data.filter(pl.col(col).is_not_null())
                if trader:
                    num_tasks = num_tasks.filter(pl.col("trader") == trader)
                annotator_tasks.append(num_tasks.shape[0])

        tasks_df = pl.DataFrame({"num_tasks": annotator_tasks})
        final = pl.concat([final, tasks_df], how="horizontal")

        trader_value = trader if trader is not None else "Total"
        final = final.with_columns(pl.lit(trader_value).alias("trader"))

        return final

    def _sum_up_per_label_metrics(
        self,
        result: dict[str, dict[str, dict[str, float]]],
        data: pl.DataFrame,
        trader: str | None = None,
    ) -> pl.DataFrame:
        """Create summary DataFrame for per-label or per-field agreement."""
        tables: dict[str, pl.DataFrame] = {}

        columns = LABEL_COLUMNS if self.case == "label" else FIELD_COLUMNS

        for annotator in result.keys():
            prim_annot_tasks = data.filter(pl.col(annotator).is_not_null()).shape[0]

            for top_email, inner_dict in result[annotator].items():
                if top_email == "ground_truth" and not self.common:
                    common_tasks = (
                        data.filter(pl.col(annotator).is_not_null())
                        .filter(pl.col(top_email).is_not_null())
                        .filter(pl.col("ground_truth_member") != annotator)
                        .shape[0]
                    )
                else:
                    common_tasks = (
                        data.filter(pl.col(annotator).is_not_null())
                        .filter(pl.col(top_email).is_not_null())
                        .shape[0]
                    )

                if len(inner_dict) < 3:
                    inner_dict = {key: 0.0 for key in columns}

                inner_dict = dict(inner_dict)  # Make a copy
                inner_dict["primary_annotator"] = annotator
                inner_dict["secondary_annotator"] = top_email
                inner_dict["prim_annot_tasks"] = prim_annot_tasks
                inner_dict["common_tasks"] = common_tasks

                if tables.get(annotator) is None:
                    schema = {
                        key: pl.Float64
                        for key in inner_dict.keys()
                        if key
                        not in [
                            "primary_annotator",
                            "secondary_annotator",
                            "prim_annot_tasks",
                            "common_tasks",
                        ]
                    }
                    schema["primary_annotator"] = pl.String
                    schema["secondary_annotator"] = pl.String
                    schema["prim_annot_tasks"] = pl.Int64
                    schema["common_tasks"] = pl.Int64
                    tables[annotator] = pl.DataFrame(schema=schema)

                tables[annotator].extend(
                    pl.from_dict(inner_dict).select(tables[annotator].columns)
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

        return master_table

    def _create_gt_breakdown(self, df: pl.DataFrame, filename: str) -> None:
        """Create ground truth breakdown CSV from a DataFrame."""
        gt_subdir = os.path.join(
            self.output_dir, "agreement_per_field", f"gt_breakdown_common_{self.common}"
        )
        os.makedirs(gt_subdir, exist_ok=True)

        output_path = os.path.join(gt_subdir, filename)

        gt_breakdown = (
            df.filter(pl.col("secondary_annotator") == "ground_truth")
            .with_columns(pl.col(FIELD_COLUMNS) * 5)
            .with_columns(
                pl.mean_horizontal(
                    pl.all().exclude(
                        [
                            "primary_annotator",
                            "secondary_annotator",
                            "prim_annot_tasks",
                            "common_tasks",
                            "trader",
                        ]
                    )
                ).alias("sum_contrib")
            )
        )

        print(output_path)
        gt_breakdown.write_csv(output_path, float_precision=3)

    def _create_gt_counts(self, df: pl.DataFrame, filename: str) -> None:
        """Create ground truth counts CSV from a DataFrame."""
        gt_subdir = os.path.join(
            self.output_dir, "agreement_per_label", f"gt_counts_common_{self.common}"
        )
        os.makedirs(gt_subdir, exist_ok=True)

        output_path = os.path.join(gt_subdir, filename)

        gt_counts = df.filter(pl.col("secondary_annotator") == "ground_truth")

        print(output_path)
        gt_counts.write_csv(output_path)
