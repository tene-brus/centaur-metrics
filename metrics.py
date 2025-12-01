import argparse
from collections import defaultdict

import polars as pl

from cli.utils.annotation_model import ListAnnotations


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

    denom = temp_2.shape[0]
    if denom == 0:
        return None if case is None else []

    temp_2 = temp_2.to_dicts()
    for row in temp_2:
        obj_1 = ListAnnotations.model_validate({"annotations": row[annotator]})
        obj_2 = ListAnnotations.model_validate({"annotations": row[annotator_2]})

        if case is not None:
            agreement_scores.append(obj_1.agreement(obj_2, case=case))
        else:
            agreement_scores += obj_1.agreement(obj_2, case=case)

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


def sum_up_per_label_metrics(result: dict, case: str, data_path: str, common: bool):
    data = pl.read_ndjson(data_path, infer_schema_length=8000).drop(["id"])

    tables = {}

    if case == "label":
        columns = [
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
    elif case == "field":
        columns = [
            "state_type",
            "direction",
            "exposure_change",
            "position_status",
            "optional_task_flags",
        ]

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

    return master_table


def sum_up_metrics(result: dict, data_path: str, trader: str | None = None):
    data = pl.read_ndjson(data_path, infer_schema_length=8000).drop(["id"])

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

    if trader is not None:
        final = final.with_columns(pl.lit(trader).alias("trader"))

    return final


def sum_up_metrics_per_trader(
    data: pl.DataFrame, annotators: list[str], common: bool, data_path: str
) -> pl.DataFrame:
    """Compute agreement metrics for each trader and concatenate results."""
    traders = data["trader"].unique().to_list()
    all_tables = []

    for trader in traders:
        trader_data = data.filter(pl.col("trader") == trader)

        # Skip if not enough data for this trader
        if trader_data.shape[0] == 0:
            continue

        scores = compute_all_pairwise_scores(trader_data, annotators, None, common)
        trader_table = sum_up_metrics(result=scores, data_path=data_path, trader=trader)
        all_tables.append(trader_table)

    if not all_tables:
        return pl.DataFrame()

    return pl.concat(all_tables, how="vertical")


def main():
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

    args = parser.parse_args()

    if args.per_trader and args.case is not None:
        parser.error("--per_trader cannot be used with --case")

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

    # Generate final table based on case type
    if args.per_trader:
        final_table = sum_up_metrics_per_trader(
            data, annotators, args.common, args.data_path
        )
    elif args.case:
        scores = compute_all_pairwise_scores(data, annotators, args.case, args.common)
        aggregated_result = aggregate_per_label_scores(scores, annotators)
        final_table = sum_up_per_label_metrics(
            result=aggregated_result,
            case=args.case,
            data_path=args.data_path,
            common=args.common,
        )
    else:
        scores = compute_all_pairwise_scores(data, annotators, args.case, args.common)
        final_table = sum_up_metrics(result=scores, data_path=args.data_path)

    # Write output
    output_file = f"agreement_case_{args.case}_common_{args.common}_per_trader_{args.per_trader}.csv"
    print(output_file)
    final_table.write_csv(output_file, float_precision=3)


if __name__ == "__main__":
    main()
