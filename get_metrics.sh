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

# Run unified metrics pipeline (computes all agreement types in a single pass)
python metrics_unified.py --data_path "$DATA_PATH" --output_dir "$METRICS_DIR"

# Merge CSVs
python merge_csvs.py --directory "${METRICS_DIR}/overall_agreement/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/common_True/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/gt_breakdown_common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_field/gt_breakdown_common_True/"

python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/common_True/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/gt_counts_common_False/"
python merge_csvs.py --directory "${METRICS_DIR}/agreement_per_label/gt_counts_common_True/"
