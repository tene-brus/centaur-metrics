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
        value=data_file.replace(".jsonl", "_metrics") if data_file else "metrics_output",
    )

with col2:
    case_type = st.selectbox("Agreement Case", ["overall", "field", "label"])
    common = st.checkbox(
        "For GT calculations, take into account common tasks where annotator's is classified as GT",
        value=False,
    )
    per_trader = st.checkbox("Per trader breakdown", value=False)

# Full output path in app/data
output_dir = str(DATA_DIR / output_dir_name)

st.markdown("---")


def build_command(
    data_file: str, output_dir: str, case_type: str, common: bool, per_trader: bool
) -> list[str]:
    # data_file is just the filename, full path is in DATA_DIR
    data_path = str(DATA_DIR / data_file)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "cli" / "metrics.py"),
        "--data_path",
        data_path,
        "--output_dir",
        output_dir,
    ]
    if case_type != "overall":
        cmd.extend(["--case", case_type])
    if common:
        cmd.append("--common")
    if per_trader:
        cmd.append("--per_trader")
    return cmd


cmd = build_command(data_file, output_dir, case_type, common, per_trader)
st.code(" ".join(cmd), language="bash")

st.info(f"Output will be saved to: `{output_dir}`")

# Run button
if st.button("Run Metrics", type="primary"):
    with st.spinner("Running metrics calculation..."):
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                st.success("Metrics calculated successfully!")
                if result.stdout:
                    st.text("Output files:")
                    for line in result.stdout.strip().split("\n"):
                        st.text(f"  {line}")
            else:
                st.error("Error running metrics")
                if result.stderr:
                    st.code(result.stderr, language="text")
        except subprocess.TimeoutExpired:
            st.error("Calculation timed out after 5 minutes")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")

# Run all combinations
st.subheader("Run All Combinations")
st.markdown(
    "Run all metric types (overall, field, label) with and without common filter, including per-trader breakdowns."
)

if st.button("Run All", type="secondary"):
    # (case, common, per_trader)
    combinations = [
        # Total agreements
        ("overall", False, False),
        ("field", False, False),
        ("field", True, False),
        ("label", False, False),
        ("label", True, False),
        # Per trader
        ("overall", False, True),
        ("field", False, True),
        ("field", True, True),
        ("label", False, True),
        ("label", True, True),
    ]

    progress = st.progress(0)
    status = st.empty()

    for i, (case, common_flag, per_trader_flag) in enumerate(combinations):
        status.text(f"Running: case={case}, common={common_flag}, per_trader={per_trader_flag}")
        cmd = build_command(data_file, output_dir, case, common_flag, per_trader_flag)

        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(result)
                st.warning(f"Warning: case={case}, common={common_flag}, per_trader={per_trader_flag} had errors")
        except Exception as e:
            st.warning(f"Error running case={case}, common={common_flag}, per_trader={per_trader_flag}: {e}")

        progress.progress((i + 1) / len(combinations))

    status.text("Complete!")
    st.success("All combinations finished!")
