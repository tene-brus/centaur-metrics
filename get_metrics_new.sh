#!/bin/bash

DATA_PATH="trade_extraction_signal1_b.jsonl"
# Extract base name without extension for metrics directory
METRICS_DIR="${DATA_PATH%.jsonl}_metrics"

source .venv/bin/activate

# total agreements
python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR"

python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "field"
python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "field" --common

python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "label"
python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "label" --common

# per trader
python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --per_trader

python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "field" --per_trader
python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "field" --common --per_trader

python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "label" --per_trader
python -m src.cli.metrics --data_path "$DATA_PATH" --output_dir "$METRICS_DIR" --case "label" --common --per_trader

# get merges
python merge_csvs.py --directory "${METRICS_DIR}/overall_agreement/common_False/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/gt_breakdown_common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/gt_breakdown_common_True/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/common_True/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/gt_counts_common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/gt_counts_common_True/"
