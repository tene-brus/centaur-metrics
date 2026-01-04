"""Tests for src/metrics/unified_pipeline.py."""

import os
import tempfile

import polars as pl
import pytest

from src.metrics.unified_pairwise import AggregatedScores, AllPairScores
from src.metrics.unified_pipeline import UnifiedMetricsPipeline
from src.models.constants import AGREEMENT_FIELDS, FIELD_COLUMNS, LABEL_COLUMNS


class TestUnifiedMetricsPipeline:
    """Tests for UnifiedMetricsPipeline class."""

    @pytest.fixture
    def sample_jsonl_file(self, tmp_path):
        """Create a sample JSONL file for testing."""
        import json

        annotation1 = {
            "label_type": "action",
            "asset_reference_type": "Majors",
            "direction": "Long",
            "action_exposure_change": "Increase",
            "action_position_status": "Clearly a new position",
        }
        annotation2 = {
            "label_type": "state",
            "asset_reference_type": "DeFi",
            "direction": "Short",
            "state_type": "Explicit State",
            "remaining_exposure": "Some",
            "state_exposure_change": "No Change",
            "state_position_status": "Clearly an existing position",
        }

        data = [
            {
                "task_id": "task1",
                "trader": "trader1",
                "num_annotations": 2,
                "predictions": [annotation1],
                "annotator1@test.com": [annotation1],
                "annotator2@test.com": [annotation1],
            },
            {
                "task_id": "task2",
                "trader": "trader1",
                "num_annotations": 2,
                "predictions": [annotation2],
                "annotator1@test.com": [annotation2],
                "annotator2@test.com": [annotation2],
            },
        ]

        file_path = tmp_path / "test_data.jsonl"
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        return str(file_path)

    @pytest.fixture
    def multi_trader_jsonl_file(self, tmp_path):
        """Create a JSONL file with multiple traders."""
        import json

        annotation1 = {
            "label_type": "action",
            "asset_reference_type": "Majors",
            "direction": "Long",
            "action_exposure_change": "Increase",
            "action_position_status": "Clearly a new position",
        }
        annotation2 = {
            "label_type": "state",
            "asset_reference_type": "DeFi",
            "direction": "Short",
            "state_type": "Explicit State",
            "remaining_exposure": "Some",
            "state_exposure_change": "No Change",
            "state_position_status": "Clearly an existing position",
        }

        data = [
            {
                "task_id": "task1",
                "trader": "trader1",
                "num_annotations": 2,
                "predictions": [annotation1],
                "annotator1@test.com": [annotation1],
                "annotator2@test.com": [annotation1],
            },
            {
                "task_id": "task2",
                "trader": "trader2",
                "num_annotations": 2,
                "predictions": [annotation2],
                "annotator1@test.com": [annotation2],
                "annotator2@test.com": [annotation2],
            },
        ]

        file_path = tmp_path / "multi_trader_data.jsonl"
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        return str(file_path)

    def test_init_with_default_output_dir(self, sample_jsonl_file):
        """Pipeline uses default output dir based on data path."""
        pipeline = UnifiedMetricsPipeline(data_path=sample_jsonl_file)

        assert pipeline.output_dir == "test_data_metrics"

    def test_init_with_custom_output_dir(self, sample_jsonl_file, tmp_path):
        """Pipeline uses custom output dir when provided."""
        output_dir = str(tmp_path / "custom_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        assert pipeline.output_dir == output_dir

    def test_run_creates_output_directory(self, sample_jsonl_file, tmp_path):
        """Pipeline creates output directory structure."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, "overall_agreement"))
        assert os.path.exists(
            os.path.join(output_dir, "agreement_per_field", "common_False")
        )
        assert os.path.exists(
            os.path.join(output_dir, "agreement_per_field", "common_True")
        )
        assert os.path.exists(
            os.path.join(output_dir, "agreement_per_label", "common_False")
        )
        assert os.path.exists(
            os.path.join(output_dir, "agreement_per_label", "common_True")
        )

    def test_run_creates_overall_agreement_csv(self, sample_jsonl_file, tmp_path):
        """Pipeline creates overall agreement CSV."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        overall_dir = os.path.join(output_dir, "overall_agreement")
        csv_files = [f for f in os.listdir(overall_dir) if f.endswith(".csv")]
        assert len(csv_files) > 0

    def test_run_creates_per_field_csv(self, sample_jsonl_file, tmp_path):
        """Pipeline creates per-field agreement CSV."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        per_field_dir = os.path.join(output_dir, "agreement_per_field", "common_False")
        csv_files = [f for f in os.listdir(per_field_dir) if f.endswith(".csv")]
        assert len(csv_files) > 0

    def test_run_creates_per_label_csv(self, sample_jsonl_file, tmp_path):
        """Pipeline creates per-label agreement CSV."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        per_label_dir = os.path.join(output_dir, "agreement_per_label", "common_False")
        csv_files = [f for f in os.listdir(per_label_dir) if f.endswith(".csv")]
        assert len(csv_files) > 0

    def test_run_creates_gt_breakdown(self, sample_jsonl_file, tmp_path):
        """Pipeline creates ground truth breakdown CSV."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        gt_breakdown_dir = os.path.join(
            output_dir, "agreement_per_field", "gt_breakdown_common_False"
        )
        assert os.path.exists(gt_breakdown_dir)

    def test_run_creates_gt_counts(self, sample_jsonl_file, tmp_path):
        """Pipeline creates ground truth counts CSV."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        gt_counts_dir = os.path.join(
            output_dir, "agreement_per_label", "gt_counts_common_False"
        )
        assert os.path.exists(gt_counts_dir)

    def test_run_total_only(self, sample_jsonl_file, tmp_path):
        """Pipeline can run total-only mode."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=False)

        overall_dir = os.path.join(output_dir, "overall_agreement")
        csv_files = [f for f in os.listdir(overall_dir) if f.endswith(".csv")]
        assert len(csv_files) == 1
        assert "Total_agreement.csv" in csv_files

    def test_per_trader_creates_separate_files(self, multi_trader_jsonl_file, tmp_path):
        """Pipeline creates separate files for each trader."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=multi_trader_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        overall_dir = os.path.join(output_dir, "overall_agreement")
        csv_files = [f for f in os.listdir(overall_dir) if f.endswith(".csv")]

        # Should have files for both traders
        assert any("trader1" in f for f in csv_files)
        assert any("trader2" in f for f in csv_files)


class TestCreateOverallDf:
    """Tests for _create_overall_df method."""

    @pytest.fixture
    def mock_all_scores(self):
        """Create mock AllPairScores for testing."""
        scores = {
            "annotator1": {
                "annotator1": None,
                "annotator2": AggregatedScores(
                    overall=0.9,
                    per_field={field: 0.2 for field in AGREEMENT_FIELDS},
                    per_label_ratios={},
                    per_label_counts={},
                    num_tasks=5,
                ),
            },
            "annotator2": {
                "annotator1": AggregatedScores(
                    overall=0.9,
                    per_field={field: 0.2 for field in AGREEMENT_FIELDS},
                    per_label_ratios={},
                    per_label_counts={},
                    num_tasks=5,
                ),
                "annotator2": None,
            },
        }
        return AllPairScores(scores=scores, annotators=["annotator1", "annotator2"])

    @pytest.fixture
    def mock_data(self):
        """Create mock data DataFrame."""
        return pl.DataFrame(
            {
                "task_id": ["task1", "task2"],
                "trader": ["trader1", "trader1"],
                "annotator1": [[{"test": "data"}], [{"test": "data"}]],
                "annotator2": [[{"test": "data"}], None],
            }
        )

    def test_creates_dataframe(self, sample_jsonl_file, tmp_path):
        """_create_overall_df returns a DataFrame."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        # Run to compute scores
        pipeline.run(per_trader=False)

        # Check output file exists and is valid
        overall_file = os.path.join(
            output_dir, "overall_agreement", "Total_agreement.csv"
        )
        df = pl.read_csv(overall_file)

        assert isinstance(df, pl.DataFrame)
        assert "annotator" in df.columns
        assert "mean_agreement" in df.columns
        assert "num_tasks" in df.columns
        assert "trader" in df.columns


class TestCreatePerFieldDf:
    """Tests for _create_per_field_df method."""

    def test_creates_per_field_dataframe(self, sample_jsonl_file, tmp_path):
        """_create_per_field_df returns DataFrame with expected columns."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        per_field_dir = os.path.join(output_dir, "agreement_per_field", "common_False")
        csv_files = [f for f in os.listdir(per_field_dir) if f.endswith(".csv")]

        for csv_file in csv_files:
            df = pl.read_csv(os.path.join(per_field_dir, csv_file))

            assert "primary_annotator" in df.columns
            assert "secondary_annotator" in df.columns
            assert "prim_annot_tasks" in df.columns
            assert "common_tasks" in df.columns
            assert "trader" in df.columns

            # Check field columns exist
            for field in FIELD_COLUMNS:
                assert field in df.columns


class TestCreatePerLabelDf:
    """Tests for _create_per_label_df method."""

    def test_creates_per_label_dataframe(self, sample_jsonl_file, tmp_path):
        """_create_per_label_df returns DataFrame with expected columns."""
        output_dir = str(tmp_path / "metrics_output")
        pipeline = UnifiedMetricsPipeline(
            data_path=sample_jsonl_file, output_dir=output_dir
        )

        pipeline.run(per_trader=True)

        per_label_dir = os.path.join(output_dir, "agreement_per_label", "common_False")
        csv_files = [f for f in os.listdir(per_label_dir) if f.endswith(".csv")]

        for csv_file in csv_files:
            df = pl.read_csv(os.path.join(per_label_dir, csv_file))

            assert "primary_annotator" in df.columns
            assert "secondary_annotator" in df.columns
            assert "prim_annot_tasks" in df.columns
            assert "common_tasks" in df.columns
            assert "trader" in df.columns


# Use shared fixture
@pytest.fixture
def sample_jsonl_file(tmp_path):
    """Create a sample JSONL file for testing."""
    import json

    annotation1 = {
        "label_type": "action",
        "asset_reference_type": "Majors",
        "direction": "Long",
        "action_exposure_change": "Increase",
        "action_position_status": "Clearly a new position",
    }
    annotation2 = {
        "label_type": "state",
        "asset_reference_type": "DeFi",
        "direction": "Short",
        "state_type": "Explicit State",
        "remaining_exposure": "Some",
        "state_exposure_change": "No Change",
        "state_position_status": "Clearly an existing position",
    }

    data = [
        {
            "task_id": "task1",
            "trader": "trader1",
            "num_annotations": 2,
            "predictions": [annotation1],
            "annotator1@test.com": [annotation1],
            "annotator2@test.com": [annotation1],
        },
        {
            "task_id": "task2",
            "trader": "trader1",
            "num_annotations": 2,
            "predictions": [annotation2],
            "annotator1@test.com": [annotation2],
            "annotator2@test.com": [annotation2],
        },
    ]

    file_path = tmp_path / "test_data.jsonl"
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return str(file_path)
