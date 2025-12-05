"""CLI entry point for metrics computation.

Usage:
    python -m src.cli.metrics --data_path data.jsonl [--case field|label] [--common] [--per_trader]
    python src/cli/metrics.py --data_path data.jsonl [--case field|label] [--common] [--per_trader]

This is the new, cleaner version of metrics.py using the refactored architecture.
"""

import sys
from pathlib import Path

# Add project root to path (must be before any src.* or cli.* imports)
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse  # noqa: E402

import src  # noqa: E402, F401 - triggers src/__init__.py to set up paths
from src.metrics.pipeline import MetricsPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Overall agreement for all data
    python -m src.cli.metrics --data_path data.jsonl

    # Per-field agreement, per trader
    python -m src.cli.metrics --data_path data.jsonl --case field --per_trader

    # Per-label agreement with common tasks only
    python -m src.cli.metrics --data_path data.jsonl --case label --common
        """,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="trade_extraction_signal1_b.jsonl",
        help="Path to the JSONL data file",
    )

    parser.add_argument(
        "--case",
        type=str,
        required=False,
        choices=["label", "field"],
        default=None,
        help="Agreement calculation type: None=overall, 'field'=per-field, 'label'=per-label",
    )

    parser.add_argument(
        "--common",
        action="store_true",
        help="Only compare on commonly-annotated tasks",
    )

    parser.add_argument(
        "--per_trader",
        action="store_true",
        help="Generate separate CSV for each trader",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Output directory (defaults to {data_path}_metrics)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for metrics CLI."""
    args = parse_args()

    pipeline = MetricsPipeline(
        data_path=args.data_path,
        case=args.case,
        common=args.common,
        output_dir=args.output_dir,
    )

    pipeline.run(per_trader=args.per_trader)


if __name__ == "__main__":
    main()
