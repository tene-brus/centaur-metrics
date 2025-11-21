import polars as pl

from cli.utils.annotation_model import ListAnnotations

data = pl.read_ndjson(
    "trade_extraction_signal1_b.jsonl", infer_schema_length=8000
).drop(["ground_truth", "alissa@zap.xyz", "tony@zap.xyz", "ground_truth_member", "id"])

data = data.filter(pl.col("num_annotations") != 0).filter(
    pl.col("predictions").is_not_null()
)

# scores = {}
# annotators = [column for column in data.columns if "@" in column] + ["predictions"]
# for annotator in annotators:
#     scores[annotator] = {}
#     temp = data.filter(pl.col(annotator).is_not_null())
#     for annotator_2 in annotators:
#         if annotator_2 == annotator:
#             scores[annotator][annotator_2] = 1
#             continue
#         scores[annotator][annotator_2] = 0

#         temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
#             [annotator, annotator_2]
#         )

#         denom = temp_2.shape[0]
#         if denom > 0:
#             temp_2 = temp_2.to_dicts()
#             for row in temp_2:
#                 obj_1 = ListAnnotations.model_validate({"annotations": row[annotator]})
#                 obj_2 = ListAnnotations.model_validate(
#                     {"annotations": row[annotator_2]}
#                 )
#                 scores[annotator][annotator_2] += obj_1 == obj_2

#             scores[annotator][annotator_2] = scores[annotator][annotator_2] / denom

scores = {}
annotators = [column for column in data.columns if "@" in column] + ["predictions"]
for annotator in annotators:
    scores[annotator] = {}
    temp = data.filter(pl.col(annotator).is_not_null())
    for annotator_2 in annotators:
        if annotator_2 == annotator:
            scores[annotator][annotator_2] = {1}
            continue
        scores[annotator][annotator_2] = 0

        temp_2 = temp.filter(pl.col(annotator_2).is_not_null()).select(
            [annotator, annotator_2]
        )

        denom = temp_2.shape[0]
        if denom > 0:
            temp_2 = temp_2.to_dicts()
            for row in temp_2:
                obj_1 = ListAnnotations.model_validate({"annotations": row[annotator]})
                obj_2 = ListAnnotations.model_validate(
                    {"annotations": row[annotator_2]}
                )
                scores[annotator][annotator_2] += obj_1 == obj_2

            scores[annotator][annotator_2] = scores[annotator][annotator_2] / denom
