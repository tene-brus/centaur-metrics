# Human Signal Ops

Tools for calculating inter-annotator agreement metrics from Label Studio annotation projects.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Label Studio credentials:
   ```
   LABEL_STUDIO_URL=https://your-label-studio-instance.com
   LABEL_STUDIO_API_KEY=your-api-key
   ```

## Project Structure

```
human_signal_ops/
--- app/                    # Streamlit web interface
--- cli/                    # CLI utilities for fetching data
--- src/                    # Core library modules
    --- agreement/          # Agreement calculation logic
    --- cli/                # CLI entry points for src modules
    --- io/                 # Data loading and CSV utilities
    --- metrics/            # Metrics computation pipeline
    --- models/             # Data models and constants
--- combine_projects.py     # Combine metrics from multiple projects
--- merge_csvs.py           # Merge per-trader CSV files
--- *.sh                    # Shell scripts for batch operations
```

## Usage

### Option 1: Streamlit App

Run the web interface:
```bash
streamlit run app/main.py
```

Pages:
- **Fetch Projects** - Download annotation data from Label Studio
- **Run Metrics** - Calculate agreement metrics
- **Merge CSVs** - Merge per-trader results
- **Combine Projects** - Combine metrics from multiple projects
- **View Results** - Browse generated CSV files

### Option 2: Shell Scripts

#### `update_projects.sh`
Fetches the latest annotation data from Label Studio for predefined projects.
```bash
./update_projects.sh
```

#### `get_metrics.sh` (depracated) / `get_metrics_new.sh`
Runs the full metrics pipeline for a JSONL data file. Calculates:
- Overall agreement scores
- Per-field agreement breakdown
- Per-label agreement breakdown
- Per-trader metrics

Then merges all per-trader CSVs into combined files.

```bash
./get_metrics_new.sh
```

### Option 3: Individual Scripts

#### Fetch Data
```bash
python cli/get_project.py --project_name "project name" --output_dir ./data
```

#### Calculate Metrics
```bash
# Overall agreement
python -m src.cli.metrics --data_path data.jsonl --output_dir metrics/

# Per-field breakdown
python -m src.cli.metrics --data_path data.jsonl --output_dir metrics/ --case field

# Per-label breakdown
python -m src.cli.metrics --data_path data.jsonl --output_dir metrics/ --case label

# Add --per_trader flag for per-trader breakdown
```

#### Merge CSVs
Merge all per-trader CSV files in a directory into a single file:
```bash
python merge_csvs.py --directory metrics/agreement_per_field/gt_breakdown_common_False/
```

#### Combine Projects
Combine metrics from two separate project directories:
```bash
python combine_projects.py --dir1 project_a_metrics/ --dir2 project_b_metrics/ --output_dir combined/
```

## Scripts Reference

| Script | Description |
|--------|-------------|
| `cli/get_project.py` | Fetch annotation data from Label Studio API |
| `src/cli/metrics.py` | Calculate pairwise agreement metrics |
| `merge_csvs.py` | Merge per-trader CSVs into a single file |
| `combine_projects.py` | Combine metrics from two projects |
| `metrics.py` | Legacy metrics script (use `src/` instead) |

## Output Files

Metrics are saved to CSV files organized by:
- `overall_agreement/` - Overall pairwise agreement scores
- `agreement_per_field/` - Breakdown by annotation field
- `agreement_per_label/` - Breakdown by label value
- `flat/` - Flattened versions with all metrics in one directory
