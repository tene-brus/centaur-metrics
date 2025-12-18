"""Tests for src/metrics/pipeline.py."""

import json
import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.metrics.pipeline import MetricsPipeline


class TestMetricsPipeline:
    """Tests for MetricsPipeline class."""

    @pytest.fixture
    def sample_jsonl_file(self, tmp_path):
        """Create sample JSONL file for testing."""
        data = [
            {
                "task_id": "t1",
                "trader": "trader_A",
                "num_annotations": 2,
                "predictions": [{"direction": "Long"}],
                "ground_truth_member": None,
                "user1@example.com": [
                    {
                        "label_type": "action",
                        "asset_reference_type": "Majors",
                        "direction": "Long",
                        "action_exposure_change": "Increase",
                        "action_position_status": "Clearly a new position",
                    }
                ],
                "user2@example.com": [
                    {
                        "label_type": "action",
                        "asset_reference_type": "Majors",
                        "direction": "Long",
                        "action_exposure_change": "Increase",
                        "action_position_status": "Clearly a new position",
                    }
                ],
            },
            {
                "task_id": "t2",
                "trader": "trader_B",
                "num_annotations": 2,
                "predictions": [{"direction": "Short"}],
                "ground_truth_member": None,
                "user1@example.com": [
                    {
                        "label_type": "action",
                        "asset_reference_type": "DeFi",
                        "direction": "Short",
                        "action_exposure_change": "Decrease",
                        "action_position_status": "Clearly an existing position",
                    }
                ],
                "user2@example.com": [
                    {
                        "label_type": "action",
                        "asset_reference_type": "DeFi",
                        "direction": "Short",
                        "action_exposure_change": "Decrease",
                        "action_position_status": "Clearly an existing position",
                    }
                ],
            },
        ]

        jsonl_path = tmp_path / "test_data.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        return str(jsonl_path)

    def test_init_sets_default_output_dir(self, sample_jsonl_file):
        """Should set default output directory based on data filename."""
        pipeline = MetricsPipeline(sample_jsonl_file)

        assert pipeline.output_dir == "test_data_metrics"

    def test_init_uses_custom_output_dir(self, sample_jsonl_file):
        """Should use custom output directory when provided."""
        pipeline = MetricsPipeline(sample_jsonl_file, output_dir="custom_output")

        assert pipeline.output_dir == "custom_output"

    def test_init_sets_case(self, sample_jsonl_file):
        """Should set case parameter correctly."""
        pipeline = MetricsPipeline(sample_jsonl_file, case="field")
        assert pipeline.case == "field"

        pipeline = MetricsPipeline(sample_jsonl_file, case="label")
        assert pipeline.case == "label"

    def test_init_sets_common_flag(self, sample_jsonl_file):
        """Should set common flag correctly."""
        pipeline = MetricsPipeline(sample_jsonl_file, common=True)
        assert pipeline.common is True

    def test_get_output_subdir_overall(self, sample_jsonl_file):
        """Should generate correct subdir for overall case (no common suffix)."""
        pipeline = MetricsPipeline(sample_jsonl_file, case=None, common=False)

        subdir = pipeline._get_output_subdir()

        assert "overall_agreement" in subdir
        assert "common_" not in subdir  # Overall doesn't use common suffix

    def test_get_output_subdir_field(self, sample_jsonl_file):
        """Should generate correct subdir for field case."""
        pipeline = MetricsPipeline(sample_jsonl_file, case="field", common=True)

        subdir = pipeline._get_output_subdir()

        assert "agreement_per_field" in subdir
        assert "common_True" in subdir

    def test_get_output_subdir_label(self, sample_jsonl_file):
        """Should generate correct subdir for label case."""
        pipeline = MetricsPipeline(sample_jsonl_file, case="label", common=False)

        subdir = pipeline._get_output_subdir()

        assert "agreement_per_label" in subdir
        assert "common_False" in subdir

    def test_run_creates_output_dir(self, sample_jsonl_file, tmp_path):
        """Should create output directory when running."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = MetricsPipeline(sample_jsonl_file, output_dir=output_dir)

        pipeline.run(per_trader=False)

        assert os.path.exists(output_dir)

    def test_run_creates_csv_file(self, sample_jsonl_file, tmp_path):
        """Should create CSV output file."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = MetricsPipeline(sample_jsonl_file, output_dir=output_dir)

        pipeline.run(per_trader=False)

        # Check for Total_agreement.csv in the subdir (no common suffix for overall)
        subdir = os.path.join(output_dir, "overall_agreement")
        csv_path = os.path.join(subdir, "Total_agreement.csv")
        assert os.path.exists(csv_path)

    def test_run_per_trader_creates_multiple_files(self, sample_jsonl_file, tmp_path):
        """Should create separate CSV for each trader."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = MetricsPipeline(sample_jsonl_file, output_dir=output_dir)

        pipeline.run(per_trader=True)

        subdir = os.path.join(output_dir, "overall_agreement")
        files = os.listdir(subdir)

        # Should have files for trader_A and trader_B
        assert any("trader_A" in f for f in files)
        assert any("trader_B" in f for f in files)


class TestPipelineIntegration:
    """Integration tests for pipeline run methods."""

    @pytest.fixture
    def sample_jsonl_file(self, tmp_path):
        """Create sample JSONL file for testing."""
        data = [
            {
                "task_id": "t1",
                "trader": "trader_A",
                "num_annotations": 2,
                "predictions": [{"direction": "Long"}],
                "ground_truth_member": None,
                "user1@example.com": [
                    {
                        "label_type": "action",
                        "asset_reference_type": "Majors",
                        "direction": "Long",
                        "action_exposure_change": "Increase",
                        "action_position_status": "Clearly a new position",
                    }
                ],
                "user2@example.com": [
                    {
                        "label_type": "action",
                        "asset_reference_type": "Majors",
                        "direction": "Long",
                        "action_exposure_change": "Increase",
                        "action_position_status": "Clearly a new position",
                    }
                ],
            },
        ]

        jsonl_path = tmp_path / "test_data.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        return str(jsonl_path)

    def test_run_with_field_case(self, sample_jsonl_file, tmp_path):
        """Should run successfully with field case."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = MetricsPipeline(sample_jsonl_file, case="field", output_dir=output_dir)

        pipeline.run(per_trader=False)

        subdir = os.path.join(output_dir, "agreement_per_field", "common_False")
        assert os.path.exists(subdir)

    def test_run_with_label_case(self, sample_jsonl_file, tmp_path):
        """Should run successfully with label case."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = MetricsPipeline(sample_jsonl_file, case="label", output_dir=output_dir)

        pipeline.run(per_trader=False)

        subdir = os.path.join(output_dir, "agreement_per_label", "common_False")
        assert os.path.exists(subdir)

    def test_output_csv_has_data(self, sample_jsonl_file, tmp_path):
        """Output CSV should contain data."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = MetricsPipeline(sample_jsonl_file, output_dir=output_dir)

        pipeline.run(per_trader=False)

        csv_path = os.path.join(output_dir, "overall_agreement", "Total_agreement.csv")
        df = pl.read_csv(csv_path)

        assert df.shape[0] > 0
        assert "annotator" in df.columns or "trader" in df.columns
