import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Combine Projects", page_icon="➕", layout="wide")
st.title("Combine Projects")
st.markdown("---")

# Paths
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PROJECT_ROOT = APP_DIR.parent

st.markdown(
    "Combine metrics from two projects by averaging agreement scores and summing task counts. "
    "This requires merged CSV files from both projects."
)

# Find metrics directories
metrics_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith("_metrics")]

if len(metrics_dirs) < 2:
    st.warning("Need at least 2 metrics directories to combine. Run metrics for multiple projects first.")
    st.stop()

metrics_dir_names = [d.name for d in metrics_dirs]

st.subheader("Select Projects to Combine")

col1, col2 = st.columns(2)

with col1:
    dir1 = st.selectbox("First Project", metrics_dir_names, key="dir1")

with col2:
    # Filter out the first selection
    remaining_dirs = [d for d in metrics_dir_names if d != dir1]
    dir2 = st.selectbox("Second Project", remaining_dirs, key="dir2")

# Output directory
default_output = "combined_metrics"
output_dir_name = st.text_input("Output Directory Name", value=default_output)
output_dir = str(DATA_DIR / output_dir_name)

st.info(f"Output will be saved to: `{output_dir}`")

st.markdown("---")

# Check for merged files in both directories
def count_merged_files(metrics_path: Path) -> int:
    return len(list(metrics_path.rglob("merged_*.csv")))

dir1_path = DATA_DIR / dir1
dir2_path = DATA_DIR / dir2

# Find corresponding JSONL files
jsonl1_name = dir1.replace("_metrics", ".jsonl")
jsonl2_name = dir2.replace("_metrics", ".jsonl")
jsonl1_path = DATA_DIR / jsonl1_name
jsonl2_path = DATA_DIR / jsonl2_name

merged1 = count_merged_files(dir1_path)
merged2 = count_merged_files(dir2_path)

st.subheader("Merged Files Status")
col1, col2 = st.columns(2)

with col1:
    if merged1 > 0:
        st.success(f"{dir1}: {merged1} merged files")
    else:
        st.error(f"{dir1}: No merged files found")

with col2:
    if merged2 > 0:
        st.success(f"{dir2}: {merged2} merged files")
    else:
        st.error(f"{dir2}: No merged files found")

if merged1 == 0 or merged2 == 0:
    st.warning("Both projects need merged CSV files. Go to 'Merge CSVs' page first.")

st.subheader("JSONL Files Status")
col1, col2 = st.columns(2)

with col1:
    if jsonl1_path.exists():
        st.success(f"{jsonl1_name}: Found")
    else:
        st.warning(f"{jsonl1_name}: Not found")

with col2:
    if jsonl2_path.exists():
        st.success(f"{jsonl2_name}: Found")
    else:
        st.warning(f"{jsonl2_name}: Not found")

if not jsonl1_path.exists() or not jsonl2_path.exists():
    st.warning("JSONL files are needed for correct task counts in per-trader rows.")

st.markdown("---")

# Combine button
if st.button("Combine Projects", type="primary", disabled=(merged1 == 0 or merged2 == 0)):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "combine_projects.py"),
        "--dir1",
        str(dir1_path),
        "--dir2",
        str(dir2_path),
        "--output_dir",
        output_dir,
    ]

    # Add JSONL paths if they exist
    jsonl_paths = []
    if jsonl1_path.exists():
        jsonl_paths.append(str(jsonl1_path))
    if jsonl2_path.exists():
        jsonl_paths.append(str(jsonl2_path))

    if jsonl_paths:
        cmd.extend(["--jsonl_paths"] + jsonl_paths)

    st.code(" ".join(cmd), language="bash")

    with st.spinner("Combining projects..."):
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                st.success("Projects combined successfully!")
                if result.stdout:
                    st.text("Output files:")
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            st.text(f"  {line}")
            else:
                st.error("Error combining projects")
                if result.stderr:
                    st.code(result.stderr, language="text")
        except subprocess.TimeoutExpired:
            st.error("Combination timed out after 2 minutes")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")

# Show existing combined directories
st.subheader("Existing Combined Outputs")
combined_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and "combined" in d.name.lower()]

if combined_dirs:
    for d in sorted(combined_dirs):
        csv_count = len(list(d.rglob("*.csv")))
        st.text(f"  {d.name} ({csv_count} CSV files)")
else:
    st.text("No combined outputs found yet.")
