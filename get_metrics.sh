#!/bin/bash

# if [ -z "$1" ]; then
#     echo "Usage: ./get_metrics.sh <jsonl_filename>"
#     echo "Example: ./get_metrics.sh trade_extraction_signal1_b.jsonl"
#     exit 1
# fi

DATA_PATH="trade_extraction_signal1_a.jsonl"
# Extract base name without extension for metrics directory
METRICS_DIR="${DATA_PATH%.jsonl}_metrics"

source .venv/bin/activate

# total agreements
python metrics.py --data_path "$DATA_PATH"

python metrics.py --data_path "$DATA_PATH" --case "field"
python metrics.py --data_path "$DATA_PATH" --case "field" --common

python metrics.py --data_path "$DATA_PATH" --case "label"
python metrics.py --data_path "$DATA_PATH" --case "label" --common

# per trader
python metrics.py --data_path "$DATA_PATH" --per_trader

python metrics.py --data_path "$DATA_PATH" --case "field" --per_trader
python metrics.py --data_path "$DATA_PATH" --case "field" --common --per_trader

python metrics.py --data_path "$DATA_PATH" --case "label" --per_trader
python metrics.py --data_path "$DATA_PATH" --case "label" --common --per_trader

# get merges
python merge_csvs.py --directory "${METRICS_DIR}/overall_agreement/common_False/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/gt_breakdown_common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/gt_breakdown_common_True/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/common_True/"
