import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Merge CSVs", page_icon="🔀", layout="wide")
st.title("Merge Per-Trader CSVs")
st.markdown("---")

# Paths
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PROJECT_ROOT = APP_DIR.parent

st.markdown(
    "Merge all per-trader CSV files in each subdirectory into a single merged CSV file."
)

# Find metrics directories
metrics_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith("_metrics")]

if not metrics_dirs:
    st.warning("No metrics directories found in app/data. Run metrics first.")
    st.stop()

# Select metrics directory
merge_dir = st.selectbox(
    "Select Metrics Directory",
    [d.name for d in metrics_dirs],
)

metrics_path = DATA_DIR / merge_dir

# Directories to merge (based on get_metrics.sh)
MERGE_SUBDIRS = [
    "overall_agreement/common_False",
    "agreement_per_field/gt_breakdown_common_False",
    "agreement_per_field/gt_breakdown_common_True",
    "agreement_per_label/common_False",
    "agreement_per_label/common_True",
    "agreement_per_label/gt_counts_common_False",
    "agreement_per_label/gt_counts_common_True",
]

st.markdown("---")

# Show which directories exist
st.subheader("Available Subdirectories")
existing_subdirs = []
for subdir in MERGE_SUBDIRS:
    full_path = metrics_path / subdir
    if full_path.exists():
        csv_count = len(list(full_path.glob("*.csv")))
        st.text(f"  ✓ {subdir} ({csv_count} CSV files)")
        existing_subdirs.append(subdir)
    else:
        st.text(f"  ✗ {subdir} (not found)")

st.markdown("---")

# Merge button
if st.button("Merge All", type="primary"):
    if not existing_subdirs:
        st.error("No subdirectories to merge.")
    else:
        progress = st.progress(0)
        status = st.empty()
        merged_count = 0

        for i, subdir in enumerate(existing_subdirs):
            full_path = metrics_path / subdir
            status.text(f"Merging: {subdir}")

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "merge_csvs.py"),
                "--directory",
                str(full_path),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    merged_count += 1
                    if result.stdout:
                        st.text(f"  Created: {result.stdout.strip()}")
                else:
                    st.warning(f"Warning: {subdir} merge failed")
                    if result.stderr:
                        st.code(result.stderr, language="text")
            except Exception as e:
                st.warning(f"Error merging {subdir}: {e}")

            progress.progress((i + 1) / len(existing_subdirs))

        status.text("Complete!")
        st.success(f"Merged {merged_count} directories!")

st.markdown("---")

# Show existing merged files
st.subheader("Existing Merged Files")
merged_files = list(metrics_path.rglob("merged_*.csv"))

if merged_files:
    for f in sorted(merged_files):
        rel_path = f.relative_to(metrics_path)
        size_kb = f.stat().st_size / 1024
        st.text(f"  {rel_path} ({size_kb:.1f} KB)")
else:
    st.text("No merged files found yet.")
