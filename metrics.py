import argparse
import os
from collections import defaultdict

import polars as pl

from cli.utils.annotation_model import ListAnnotations

LABEL_COLUMNS = [
    "action",
    "state",
    "Specific Asset(s)",
    "Majors",
    "DeFi",
    "AI",
    "AI Agents",
    "RWA",
    "Layer 1",
    "Layer 2",
    "Alts",
    "All Open Positions",
    "All Long Positions",
    "All Shorts",
    "Memes",
    "Other",
    "Long",
    "Short",
    "Unclear",
    "Clearly a new position",
    "Clearly an existing position",
    "Increase",
    "Decrease",
    "No Change",
    "Some",
    "None",
    "Explicit State",
    "Direct State",
    "Indirect State",
]

FIELD_COLUMNS = [
    "state_type",
    "direction",
    "exposure_change",
    "position_status",
    "optional_task_flags",
]


def calculate_pairwise_agreement(
    data: pl.DataFrame,
    annotator: str,
    annotator_2: str,
    case: str | None,
    common: bool,
) -> list | float | None:
    """Calculate agreement between two annotators."""
    temp = data.filter(pl.col(annotator).is_not_null())

    if case:
        agreement_scores = []
    else:
        agreement_scores = 0.0

    # Calculate agreement with ground truth when the annotator is not the ground truth member
    if "ground_truth" in [annotator, annotator_2] and not common:
        temp_2 = (
            temp.filter(pl.col(annotator_2).is_not_null())
            .filter(~pl.col("ground_truth_member").is_in([annotator, annotator_2]))
            .select([annotator, annotator_2])
        )
    else:
        temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
            [annotator, annotator_2]
        )

    if temp_2.shape[0] == 0:
        return None if case is None else []

    temp_2 = temp_2.to_dicts()
    denom = 0
    for row in temp_2:
        obj_1 = ListAnnotations.model_validate({"annotations": row[annotator]})
        obj_2 = ListAnnotations.model_validate({"annotations": row[annotator_2]})

        if case == "label":
            # Skip rows where both annotators have empty annotations
            if obj_1.annotations == [] and obj_2.annotations == []:
                continue
            agreement_scores.append(obj_1.agreement(obj_2, case=case))
            denom += 1
        elif case == "field":
            agreement_scores.append(obj_1.agreement(obj_2, case=case))
            denom += 1
        else:
            agreement_scores += obj_1.agreement(obj_2, case=case)
            denom += 1

    if denom == 0:
        return None if case is None else []

    if case is None:
        agreement_scores = agreement_scores / denom

    return agreement_scores


def compute_all_pairwise_scores(
    data: pl.DataFrame, annotators: list[str], case: str | None, common: bool
) -> dict:
    """Compute agreement scores between all pairs of annotators."""
    scores = {}
    for annotator in annotators:
        scores[annotator] = {}
        for annotator_2 in annotators:
            if annotator_2 == annotator:
                scores[annotator][annotator_2] = None
                continue

            scores[annotator][annotator_2] = calculate_pairwise_agreement(
                data, annotator, annotator_2, case, common
            )

    return scores


def aggregate_per_label_scores(scores: dict, annotators: list[str]) -> dict:
    """Aggregate per-label scores into mean values."""
    result = {}
    for annotator in annotators:
        result[annotator] = {}
        for annotator_2 in annotators:
            if annotator_2 == annotator:
                continue

            result[annotator][annotator_2] = defaultdict(float)

            n = len(scores[annotator][annotator_2])
            for item in scores[annotator][annotator_2]:
                for key, value in item.items():
                    result[annotator][annotator_2][key] += value

            # Convert to mean
            result[annotator][annotator_2] = {
                key: total / n for key, total in result[annotator][annotator_2].items()
            }

    return result


def sum_up_per_label_metrics(
    result: dict, case: str, data: pl.DataFrame, common: bool, trader: str | None = None
) -> pl.DataFrame:
    tables = {}

    if case == "label":
        columns = LABEL_COLUMNS
    elif case == "field":
        columns = FIELD_COLUMNS

    for annotator in result.keys():
        # get annotator number of tasks
        prim_annot_tasks = data.filter(pl.col(annotator).is_not_null()).shape[0]
        for top_email, inner_dict in result[annotator].items():
            if top_email == "ground_truth" and not common:
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
    for annotator in tables.keys():
        master_table.extend(tables[annotator].select(master_table.columns))

    trader_value = trader if trader is not None else "Total"
    master_table = master_table.with_columns(pl.lit(trader_value).alias("trader"))

    return master_table


def sum_up_metrics(result: dict, data: pl.DataFrame, trader: str | None = None) -> pl.DataFrame:
    df = pl.DataFrame(schema={key: pl.Float64 for key in result.keys()})

    for annotator in result.keys():
        df = df.extend(pl.from_dict(result[annotator]))

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


def compute_metrics_for_trader(
    data: pl.DataFrame,
    annotators: list[str],
    case: str | None,
    common: bool,
    trader: str | None = None,
) -> pl.DataFrame:
    """Compute agreement metrics for a single trader (or all data if trader is None)."""
    scores = compute_all_pairwise_scores(data, annotators, case, common)

    if case:
        aggregated_result = aggregate_per_label_scores(scores, annotators)
        return sum_up_per_label_metrics(
            result=aggregated_result,
            case=case,
            data=data,
            common=common,
            trader=trader,
        )
    else:
        return sum_up_metrics(result=scores, data=data, trader=trader)


def get_output_subdir(output_dir: str, case: str | None, common: bool) -> str:
    """Get the output subdirectory path based on case and common flags."""
    if case is None:
        case_subdir = "overall_agreement"
    else:
        case_subdir = f"agreement_per_{case}"
    return os.path.join(output_dir, case_subdir, f"common_{common}")


def run_per_trader(
    data: pl.DataFrame,
    annotators: list[str],
    case: str | None,
    common: bool,
    output_dir: str,
) -> None:
    """Run metrics computation for each trader and save separate CSVs."""
    subdir = get_output_subdir(output_dir, case, common)
    os.makedirs(subdir, exist_ok=True)

    traders = data["trader"].unique().to_list()

    for trader in traders:
        trader_data = data.filter(pl.col("trader") == trader)

        if trader_data.shape[0] == 0:
            continue

        trader_table = compute_metrics_for_trader(
            trader_data, annotators, case, common, trader
        )

        filename = f"agreement_{trader}.csv"
        output_file = os.path.join(subdir, filename)
        print(output_file)

        if case:
            trader_table.write_csv(output_file, float_precision=3)
            # Generate ground truth breakdown for field case
            if case == "field":
                create_gt_breakdown(trader_table, output_dir, common, filename)
        else:
            trader_table.filter(pl.col("annotator").is_not_null()).write_csv(
                output_file, float_precision=3
            )


def create_gt_breakdown(df: pl.DataFrame, output_dir: str, common: bool, filename: str) -> None:
    """Create ground truth breakdown CSV from a DataFrame."""
    gt_subdir = os.path.join(
        output_dir, "agreement_per_field", f"gt_breakdown_common_{common}"
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


def main() -> None:
    """Main entry point for computing inter-annotator agreement metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="trade_extraction_signal1_b.jsonl",
    )
    parser.add_argument(
        "--case",
        type=str,
        required=False,
        choices=["label", "field"],
        default=None,
    )
    parser.add_argument("--common", action="store_true")
    parser.add_argument("--per_trader", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Output directory for per-trader CSVs (defaults to {data_path}_metrics)",
    )

    args = parser.parse_args()

    # Set output_dir based on data_path if not provided
    if args.output_dir is None:
        base_name = os.path.splitext(os.path.basename(args.data_path))[0]
        args.output_dir = f"{base_name}_metrics"

    # Load and filter data
    data = pl.read_ndjson(args.data_path, infer_schema_length=8000).drop(["id"])
    data = data.filter(pl.col("num_annotations") != 0).filter(
        pl.col("predictions").is_not_null()
    )

    # Get list of annotators from columns
    annotators = [column for column in data.columns if "@" in column] + [
        "predictions",
        "ground_truth",
    ]

    # Generate output based on mode
    if args.per_trader:
        run_per_trader(data, annotators, args.case, args.common, args.output_dir)
    else:
        subdir = get_output_subdir(args.output_dir, args.case, args.common)
        os.makedirs(subdir, exist_ok=True)

        final_table = compute_metrics_for_trader(
            data, annotators, args.case, args.common
        )
        filename = "Total_agreement.csv"
        output_file = os.path.join(subdir, filename)
        print(output_file)

        if args.case == "field":
            final_table.write_csv(output_file, float_precision=3)
            create_gt_breakdown(final_table, args.output_dir, args.common, filename)
        elif args.case == "label":
            final_table.write_csv(output_file, float_precision=3)
        else:
            # case is None (overall agreement)
            final_table.filter(pl.col("annotator").is_not_null()).write_csv(
                output_file, float_precision=3
            )


if __name__ == "__main__":
    main()
