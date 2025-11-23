import argparse
from collections import defaultdict

import polars as pl

from cli.utils.annotation_model import ListAnnotations


def sum_up_per_label_metrics(result: dict):
    tables = {}

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

    for annotator in result.keys():
        for top_email, inner_dict in result[annotator].items():
            if len(inner_dict) < 3:
                inner_dict = {key: 0.0 for key in columns}

            inner_dict["primary_annotator"] = annotator
            inner_dict["secondary_annotator"] = top_email

            if tables.get(annotator) is None:
                schema = {
                    key: pl.Float64
                    for key in inner_dict.keys()
                    if key not in ["primary_annotator", "secondary_annotator"]
                }
                schema["primary_annotator"] = pl.String
                schema["secondary_annotator"] = pl.String
                tables[annotator] = pl.DataFrame(schema=schema)

            tables[annotator].extend(pl.from_dict(inner_dict))

    schema = {key: pl.Float64 for key in columns}
    schema["primary_annotator"] = pl.String
    schema["secondary_annotator"] = pl.String
    master_table = pl.DataFrame(schema=schema)
    for annotator in tables.keys():
        master_table.extend(tables[annotator])

    return master_table


def sum_up_metrics(result: dict):
    df = pl.DataFrame(schema={key: pl.Float64 for key in result.keys()})

    for annotator in result.keys():
        df = df.extend(pl.from_dict(result[annotator]))

    df = df.with_columns(pl.mean_horizontal(pl.all()).alias("mean_agreement"))

    col_df = pl.DataFrame({"annotator": df.columns[:-1]})

    return pl.concat([col_df, df], how="horizontal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--per_label", type=str, required=False)

    args = parser.parse_args()

    args.data_path = "trade_extraction_signal1_b.jsonl"
    args.per_label = True

    data = pl.read_ndjson(args.data_path, infer_schema_length=8000).drop(
        ["ground_truth", "ground_truth_member", "id"]
    )

    data = data.filter(pl.col("num_annotations") != 0).filter(
        pl.col("predictions").is_not_null()
    )

    scores = {}
    annotators = [column for column in data.columns if "@" in column] + ["predictions"]
    for annotator in annotators:
        scores[annotator] = {}
        temp = data.filter(pl.col(annotator).is_not_null())
        for annotator_2 in annotators:
            if annotator_2 == annotator:
                scores[annotator][annotator_2] = None
                continue

            if args.per_label:
                scores[annotator][annotator_2] = []
            else:
                scores[annotator][annotator_2] = 0.0

            temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
                [annotator, annotator_2]
            )

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
                    if args.per_label:
                        scores[annotator][annotator_2].append(
                            obj_1.agreement(obj_2, per_label=args.per_label)
                        )
                    else:
                        scores[annotator][annotator_2] += obj_1.agreement(obj_2)

                if not args.per_label:
                    scores[annotator][annotator_2] = (
                        scores[annotator][annotator_2] / denom
                    )
            elif denom == 0 and not args.per_label:
                scores[annotator][annotator_2] = None

    if args.per_label:
        result = {}
        for annotator in annotators:
            result[annotator] = {}
            for annotator_2 in annotators:
                if annotator_2 == annotator:
                    continue

                result[annotator][annotator_2] = defaultdict(int)

                n = len(scores[annotator][annotator_2])
                for item in scores[annotator][annotator_2]:
                    for key, value in item.items():
                        result[annotator][annotator_2][key] += value

                # Convert to mean
                result[annotator][annotator_2] = {
                    key: total / n
                    for key, total in result[annotator][annotator_2].items()
                }

        final_table = sum_up_per_label_metrics(result=result)
    else:
        final_table = sum_up_metrics(result=scores)

    final_table.write_csv(
        f"agreement_per_label_{args.per_label}.csv", float_precision=3
    )
