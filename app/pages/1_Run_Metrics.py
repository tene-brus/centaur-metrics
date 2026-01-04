import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Run Metrics", page_icon="🚀", layout="wide")
st.title("Run Metrics")
st.markdown("---")

# Paths
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PROJECT_ROOT = APP_DIR.parent

# Find available JSONL files in app/data
jsonl_files = list(DATA_DIR.glob("*.jsonl"))
jsonl_names = [f.name for f in jsonl_files]

if not jsonl_names:
    st.warning("No JSONL files found in app/data. Fetch projects first.")
    st.stop()

# Configuration
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    data_file = st.selectbox("Data File", jsonl_names)
    output_dir_name = st.text_input(
        "Output Directory Name",
        value=data_file.replace(".jsonl", "_metrics")
        if data_file
        else "metrics_output",
    )

# Full paths
data_path = str(DATA_DIR / data_file)
output_dir = str(DATA_DIR / output_dir_name)

st.markdown("---")

# Unified pipeline
st.subheader("Run All Metrics")
st.markdown(
    """
    Computes **all agreement types** (overall, per-field, per-label) in a single optimized pass.
    """
)

st.info(f"Output will be saved to: `{output_dir}`")

if st.button("Run Metrics", type="primary"):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "metrics_unified.py"),
        "--data_path",
        data_path,
        "--output_dir",
        output_dir,
    ]

    st.code(" ".join(cmd), language="bash")

    with st.spinner("Running metrics calculation..."):
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                st.success("Metrics calculated successfully!")
                if result.stdout:
                    # Count output files
                    lines = [
                        ln for ln in result.stdout.strip().split("\n") if ln.strip()
                    ]
                    st.text(f"Generated {len(lines)} output files")
                    with st.expander("Show output files"):
                        for line in lines:
                            st.text(f"  {line}")
            else:
                st.error("Error running metrics")
                if result.stderr:
                    st.code(result.stderr, language="text")
        except subprocess.TimeoutExpired:
            st.error("Calculation timed out after 10 minutes")
        except Exception as e:
            st.error(f"Error: {e}")
