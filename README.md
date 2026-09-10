# Centaur Metrics

Tools for calculating inter-annotator agreement metrics from Label Studio annotation projects.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python >= 3.12.

1. Install dependencies:
   ```bash
   uv sync
   ```
   Add `--group dev` (or `uv sync --all-groups`) if you also want `pytest` and `pre-commit`.

2. Create a `.env` file with your Label Studio credentials:
   ```bash
   cp .env.example .env   # then fill in the two values
   ```
   | Variable | Description |
   |----------|-------------|
   | `LABEL_STUDIO_URL` | Base URL of your Label Studio instance |
   | `LABEL_STUDIO_API_KEY` | API key for that instance |

   These are the only environment variables the code reads (`cli/get_project.py`).

3. (Optional) Install the pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Project Structure

```
centaur-metrics/
--- app/                    # Streamlit web interface
    --- main.py             # Dashboard entry point
    --- pages/              # One file per dashboard page
    --- data/               # JSONL data + metrics output (created at runtime)
--- cli/                    # Label Studio fetching CLI
--- src/                    # Core library modules
    --- agreement/          # Agreement calculation logic (matching, unified)
    --- io/                 # Data loading, path config and CSV utilities
    --- metrics/            # Metrics pipeline + reviewer quality
    --- models/             # Data models and constants
--- tests/                  # Pytest unit tests
--- metrics_unified.py      # Single-pass metrics pipeline entry point
--- merge_csvs.py           # Merge per-trader CSV files
--- combine_projects.py     # Combine metrics from two projects
--- get_metrics.sh          # End-to-end pipeline for one JSONL file
--- update_projects.sh      # Fetch one or more projects
--- .env.example            # Template for the required env vars
--- Dockerfile              # Streamlit app image
--- docker-compose.yaml     # Runs the app on port 8501
```

## Configuration

Beyond the two environment variables, runtime configuration lives in
`app/data/reviewer_config.json`. It is gitignored, because it holds real user
IDs and emails. The Reviewer Config page writes the file the first time you
save an exclusion; on a fresh install it does not exist yet, so create it by
hand if you need `gt_verifiers` before then.

```json
{
  "global_exclusions": ["reviewer@example.com"],
  "project_reviewers": {"project_name": ["reviewer@example.com"]},
  "gt_verifiers": {"12345": "verifier@example.com"}
}
```

| Key | Purpose | Managed by |
|-----|---------|------------|
| `global_exclusions` | Annotators excluded from metrics in every project | Reviewer Config page |
| `project_reviewers` | Annotators excluded from one project only | Reviewer Config page |
| `gt_verifiers` | Label Studio user ID -> email for whoever accepts ground truth, used to fill `gt_accepted_by` during fetch | Hand-edited |

`gt_verifiers` is read by `cli/get_project.py`. If it is missing, fetching still
works but `gt_accepted_by` is left unset, and the CLI logs a warning saying so.

### Finding Label Studio user IDs

`gt_verifiers` is keyed by numeric Label Studio user ID, which the web UI does
not show. To list the ID and email of every user on the instance:

```bash
uv run python -c "
import os
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

load_dotenv()
client = LabelStudio(
    base_url=os.getenv('LABEL_STUDIO_URL'), api_key=os.getenv('LABEL_STUDIO_API_KEY')
)
for user in client.users.list():
    print(user.id, user.email)
"
```

Take the IDs of whoever accepts ground truth and add them as JSON *string* keys
(`"73379"`, not `73379`) — the loader casts them back to ints:

```json
"gt_verifiers": {"73379": "verifier@example.com"}
```

## Usage

### Option 1: Streamlit App (Docker)

```bash
docker compose up --build
```

The app is served on <http://localhost:8501>. `.env` is read via `env_file`, and
`./app/data` is mounted into the container so fetched data and generated metrics
persist on the host.

### Option 2: Streamlit App (local)

```bash
uv run streamlit run app/main.py
```

Pages:

| Page | Description |
|------|-------------|
| **Fetch Projects** | Download annotation data from Label Studio (single project, all predefined projects, or upload an existing JSONL file) |
| **Run Metrics** | Run the unified metrics pipeline over a JSONL file |
| **Merge CSVs** | Merge per-trader CSVs in a metrics subdirectory |
| **Combine Projects** | Combine metrics from two project directories |
| **View Results** | Browse, filter and download merged CSVs, with agreement heatmaps |
| **GT Quality** | Annotator agreement with ground truth, filterable by trader and field/label |
| **Reviewer Config** | Configure global exclusions and per-project reviewers (`app/data/reviewer_config.json`) |
| **Reviewer Quality** | Verifier submission stats and reviewer error frequency |

All pages read and write under `app/data/`: JSONL files at the top level, and
one `<name>_metrics/` directory per project.

### Option 3: Shell Scripts

#### `update_projects.sh`
Fetches the latest annotation data from Label Studio for one or more projects.
```bash
./update_projects.sh "trade extraction - signal1 - A" "trade extraction - signal1 - B"
```

#### `get_metrics.sh`
Runs the full pipeline for a JSONL file: the unified metrics pass, then a merge
of every per-trader CSV directory it produces.
```bash
./get_metrics.sh trade_extraction_signal1_a.jsonl
```

### Option 4: Individual Scripts

#### Fetch Data
```bash
uv run python cli/get_project.py --project_name "project name" --output_dir ./app/data
```
`--output_dir` is optional and defaults to the current directory.

#### Calculate Metrics
One pass computes overall, per-field and per-label agreement (including the
ground-truth breakdowns), for both the `common` and non-`common` variants:
```bash
uv run python metrics_unified.py --data_path data.jsonl --output_dir metrics/
```
- `--output_dir` defaults to `<data basename>_metrics`.
- Per-trader CSVs are written by default; pass `--total_only` for
  `Total_agreement.csv` only.

#### Merge CSVs
Merge all per-trader CSV files in a directory into a single file, written as
`merged_<dirname>.csv` inside that directory:
```bash
uv run python merge_csvs.py \
  --directory metrics/agreement_per_field/gt_breakdown_common_False/ \
  --jsonl_path data.jsonl
```
`--jsonl_path` is optional and only needed to add task counts to the output.

#### Combine Projects
Combine metrics from two separate project directories:
```bash
uv run python combine_projects.py \
  --dir1 project_a_metrics/ --dir2 project_b_metrics/ \
  --output_dir combined/ --jsonl_paths project_a.jsonl project_b.jsonl
```
`--output_dir` defaults to `combined_metrics`; `--jsonl_paths` is optional and
takes one or more files.

## Deployment

The app ships as a container built from the `Dockerfile`. Any platform that can
build and run it works — Docker Compose on a VM, or a managed platform such as
Railway, Fly, Cloud Run or ECS. The requirements below are what the container
needs from its host, whichever one you pick.

**Runtime contract**

| Requirement | Detail |
|-------------|--------|
| Build | The repo root `Dockerfile`. No build args or build-time secrets. |
| Env vars | `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY`, injected at runtime. They are never baked into the image. |
| Port | The container listens on `8501` (pinned via `STREAMLIT_SERVER_PORT`). If your platform injects a `$PORT`, override that env var to match. |
| Persistent storage | A volume mounted at `/workspace/app/data`. See below — without one you lose data on every redeploy. |
| Health | `GET /` returns 200 once Streamlit is up. |

**Persistent storage.** `app/data` holds everything worth keeping: fetched
JSONL files, generated `*_metrics/` directories, and `reviewer_config.json`.
There is no database — that directory is the entire application state, and the
rest of the container is disposable. It must be a real volume; on platforms
with ephemeral filesystems, a redeploy silently wipes it otherwise. Back it up.

**Network exposure.** The app has **no authentication of its own**. Anyone who
can reach it can read every annotation in `app/data` and trigger Label Studio
fetches using the configured API key. Keep it on a private network, or put a
reverse proxy or the platform's access control in front of it.

**Reference: Docker Compose.** `docker-compose.yaml` wires all of the above up
for a single host — `env_file` for the credentials, a bind mount for
`./app/data`, and `restart: unless-stopped`. Create `.env` on the host first
(it is gitignored and never committed), then:

```bash
docker compose up -d --build          # start, and redeploy after a git pull
```

Editing `.env` needs only `docker compose up -d`; no rebuild is required.

## Scripts Reference

| Script | Description |
|--------|-------------|
| `cli/get_project.py` | Fetch annotation data from Label Studio API |
| `metrics_unified.py` | Compute all agreement metrics in a single pass |
| `merge_csvs.py` | Merge per-trader CSVs into a single file |
| `combine_projects.py` | Combine metrics from two projects |
| `update_projects.sh` | Fetch one or more Label Studio projects by name |
| `get_metrics.sh` | Metrics + merge pipeline for one JSONL file |

## Output Files

`metrics_unified.py` writes, under the output directory:

```
overall_agreement/                                  # Overall pairwise agreement
agreement_per_field/common_{True,False}/            # Breakdown by annotation field
agreement_per_field/gt_breakdown_common_{True,False}/   # Per-field agreement with GT
agreement_per_label/common_{True,False}/            # Breakdown by label value
agreement_per_label/gt_counts_common_{True,False}/  # Per-label GT counts
```

Each directory holds `Total_agreement.csv` plus one `agreement_<trader>.csv`
per trader; `merge_csvs.py` adds `merged_<dirname>.csv`. `combine_projects.py`
additionally writes a `flat/` directory with all combined files in one place.

## Tests

```bash
uv run pytest
```
