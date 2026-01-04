#!/usr/bin/env python3
"""Unified metrics computation script - computes all agreement types in a single pass.

This is ~3x faster than the original metrics.py which ran separate passes
for each agreement type (overall, per_field, per_label).

Usage:
    python metrics_unified.py --data_path path/to/data.jsonl [--output_dir metrics_output]
"""

import argparse

from src.metrics.unified_pipeline import UnifiedMetricsPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute all agreement metrics in a single pass"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input JSONL file with annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to {data_basename}_metrics)",
    )
    parser.add_argument(
        "--per_trader",
        action="store_true",
        default=True,
        help="Generate per-trader CSV files (default: True)",
    )
    parser.add_argument(
        "--total_only",
        action="store_true",
        help="Only generate Total_agreement.csv (no per-trader files)",
    )

    args = parser.parse_args()

    pipeline = UnifiedMetricsPipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

    per_trader = not args.total_only
    pipeline.run(per_trader=per_trader)


if __name__ == "__main__":
    main()
