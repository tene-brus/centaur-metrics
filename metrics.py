import argparse
from collections import defaultdict

import polars as pl

from cli.utils.annotation_model import ListAnnotations


def sum_up_per_label_metrics(result: dict, case: str):
    data = pl.read_ndjson(
        "trade_extraction_signal1_b.jsonl", infer_schema_length=8000
    ).drop(["id"])

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


def sum_up_metrics(result: dict):
    data = pl.read_ndjson(
        "trade_extraction_signal1_b.jsonl", infer_schema_length=8000
    ).drop(["id"])

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
        if col in ["annotator", "mean_agreement"]:
            continue
        num_tasks = data.filter(pl.col(col).is_not_null()).shape[0]
        annotator_tasks.append(num_tasks)

    tasks_df = pl.DataFrame({"num_tasks": annotator_tasks})

    return pl.concat([final, tasks_df], how="horizontal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument(
        "--case",
        type=str,
        required=False,
        choices=["label", "field"],
        default=None,
    )
    parser.add_argument("--common", action="store_true")

    args = parser.parse_args()

    args.data_path = "trade_extraction_signal1_b.jsonl"
    # args.case = "field"

    data = pl.read_ndjson(args.data_path, infer_schema_length=8000).drop(["id"])

    data = data.filter(pl.col("num_annotations") != 0).filter(
        pl.col("predictions").is_not_null()
    )

    scores = {}
    annotators = [column for column in data.columns if "@" in column] + [
        "predictions",
        "ground_truth",
    ]
    for annotator in annotators:
        scores[annotator] = {}
        temp = data.filter(pl.col(annotator).is_not_null())
        # annot_tasks = temp.shape[0]
        for annotator_2 in annotators:
            if annotator_2 == annotator:
                scores[annotator][annotator_2] = None
                continue

            if args.case:
                scores[annotator][annotator_2] = []
            else:
                scores[annotator][annotator_2] = 0.0

            # calculate agreement with ground truth when the annotator is not the ground truth member
            # this is how it is computed in HS
            if "ground_truth" in [annotator, annotator_2] and not args.common:
                temp_2 = (
                    temp.filter(pl.col(annotator_2).is_not_null())
                    .filter(
                        ~pl.col("ground_truth_member").is_in([annotator, annotator_2])
                    )
                    .select([annotator, annotator_2])
                )
            else:
                temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
                    [annotator, annotator_2]
                )
            # common_tasks = temp_2.shape[0]

            denom = temp_2.shape[0]
            if denom > 0:
                temp_2 = temp_2.to_dicts()
                for row in temp_2:
                    obj_1 = ListAnnotations.model_validate(
                        {"annotations": row[annotator]}
                    )
                    obj_2 = ListAnnotations.model_validate(
                        {"annotations": row[annotator_2]}
                    )
                    if args.case is not None:
                        scores[annotator][annotator_2].append(
                            obj_1.agreement(obj_2, case=args.case)
                        )
                    else:
                        scores[annotator][annotator_2] += obj_1.agreement(
                            obj_2, case=args.case
                        )

                if args.case is None:
                    scores[annotator][annotator_2] = (
                        scores[annotator][annotator_2] / denom
                    )
            elif denom == 0 and args.case is None:
                scores[annotator][annotator_2] = None

    if args.case:
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
                    key: total / n
                    for key, total in result[annotator][annotator_2].items()
                }

        final_table = sum_up_per_label_metrics(result=result, case=args.case)
    else:
        final_table = sum_up_metrics(result=scores)

    print(f"agreement_case_{args.case}_common_{str(args.common)}.csv")
    final_table.write_csv(
        f"agreement_case_{args.case}_common_{str(args.common)}.csv", float_precision=3
    )
