"""Reviewer Quality Dashboard - View reviewer error frequency per project."""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from src.metrics.reviewer_quality import calculate_reviewer_error_frequency

st.set_page_config(page_title="Reviewer Quality", page_icon="🔍", layout="wide")
st.title("Reviewer Quality Dashboard")
st.markdown(
    "View how often the reviewer's annotations were corrected (differ from final GT)."
)


def display_task_breakdown(
    label: str,
    total: int,
    reviewed: int,
    reviewed_errors: int,
    not_reviewed: int,
):
    """Display a task breakdown section with reviewed/not-reviewed stats."""
    st.markdown(f"**{label}:** {total} tasks")
    if total > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*Reviewed by Reviewer:*")
            st.metric("Total", reviewed)
            error_rate = reviewed_errors / reviewed if reviewed > 0 else 0
            st.metric("Reviewer Mistakes", f"{reviewed_errors} ({error_rate:.1%})")
        with col2:
            st.markdown("*Not Reviewed:*")
            st.metric("Total", not_reviewed)


st.markdown("---")

# Paths
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
CONFIG_PATH = DATA_DIR / "reviewer_config.json"


def load_config() -> dict:
    """Load reviewer config from file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"global_exclusions": [], "project_reviewers": {}}


# Load config
config = load_config()
project_reviewers = config.get("project_reviewers", {})
gt_verifiers = config.get("gt_verifiers", [])

# Find projects with configured reviewers
projects_with_reviewers = {
    project: reviewers[0]
    for project, reviewers in project_reviewers.items()
    if reviewers
}

if not projects_with_reviewers:
    st.warning(
        "No project reviewers configured. Go to 'Reviewer Config' page to set them up."
    )
    st.stop()

# Project selector
st.subheader("Select Project")
selected_project = st.selectbox(
    "Project",
    list(projects_with_reviewers.keys()),
    help="Select a project to view reviewer quality",
)

reviewer_email = projects_with_reviewers[selected_project]
st.info(f"**Reviewer:** {reviewer_email}")

# Load data
jsonl_path = DATA_DIR / f"{selected_project}.jsonl"

if not jsonl_path.exists():
    st.error(f"Data file not found: {jsonl_path}")
    st.stop()

# Calculate error frequency
with st.spinner("Calculating reviewer error frequency..."):
    data = pl.read_ndjson(str(jsonl_path), infer_schema_length=8000)
    result = calculate_reviewer_error_frequency(
        data, reviewer_email, selected_project, gt_verifiers=gt_verifiers
    )

if not result:
    st.error(
        "Could not calculate reviewer error frequency. Check that the reviewer has annotations and GT exists."
    )
    st.stop()

st.markdown("---")

# Project-level statistics
st.subheader("Project Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Project Tasks", result.project_total_tasks)

with col2:
    st.metric("Tasks Reviewed", result.total_tasks)

with col3:
    st.metric("Tasks Not Reviewed", result.tasks_not_reviewed)

with col4:
    st.metric("Overall Error Rate", f"{result.error_frequency:.1%}")

st.markdown("---")

# GT Verifier Own Submissions - per verifier breakdown
if result.verifier_own_submission_stats:
    st.subheader("Verifier Own Submissions")
    st.markdown("Tasks where each GT verifier submitted their own annotation as GT.")

    for verifier, stats in result.verifier_own_submission_stats.items():
        verifier_name = verifier.split("@")[0].capitalize()
        with st.expander(
            f"{verifier_name} ({verifier}) - {stats.total} tasks",
            expanded=stats.total > 0,
        ):
            if stats.total > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Reviewed by Reviewer:**")
                    st.metric("Total", stats.reviewed_total)
                    if stats.reviewed_total > 0:
                        error_rate = stats.reviewed_with_errors / stats.reviewed_total
                        st.metric(
                            "Reviewer Mistakes",
                            f"{stats.reviewed_with_errors} ({error_rate:.1%})",
                        )
                    else:
                        st.metric("Reviewer Mistakes", "0")
                with col2:
                    st.markdown("**Not Reviewed:**")
                    st.metric("Total", stats.not_reviewed_total)
            else:
                st.info("No tasks")

st.markdown("---")

# Verifier Accepted Other Annotator's Submission - per verifier breakdown
if result.verifier_accepted_stats:
    st.subheader("Verifier Accepted Other Annotator's Submission")
    st.markdown(
        "Tasks where each GT verifier accepted an existing annotator's submission as GT."
    )

    for verifier, stats in result.verifier_accepted_stats.items():
        verifier_name = verifier.split("@")[0].capitalize()
        with st.expander(
            f"{verifier_name} ({verifier}) - {stats.total} tasks",
            expanded=stats.total > 0,
        ):
            if stats.total > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Reviewed by Reviewer:**")
                    st.metric("Total", stats.reviewed_total)
                    if stats.reviewed_total > 0:
                        error_rate = stats.reviewed_with_errors / stats.reviewed_total
                        st.metric(
                            "Reviewer Mistakes",
                            f"{stats.reviewed_with_errors} ({error_rate:.1%})",
                        )
                    else:
                        st.metric("Reviewer Mistakes", "0")
                with col2:
                    st.markdown("**Not Reviewed:**")
                    st.metric("Total", stats.not_reviewed_total)
            else:
                st.info("No tasks")

st.markdown("---")

# Reviewer error summary (overall)
st.subheader("Reviewer Error Summary (Overall)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tasks Reviewed", result.total_tasks)

with col2:
    st.metric("Tasks with Errors", result.tasks_with_errors)

with col3:
    st.metric("Error Frequency", f"{result.error_frequency:.1%}")

st.markdown("---")

# Per-trader breakdown
st.subheader("Per-Trader Breakdown")
st.markdown(
    "Error frequency by trader - higher values indicate less trustworthy GT for that trader."
)

# Convert to DataFrame for display
trader_data = []
for trader, data in result.per_trader.items():
    trader_data.append(
        {
            "Trader": trader,
            "Total Tasks": data["total"],
            "Tasks with Errors": data["errors"],
            "Error Frequency": data["frequency"],
        }
    )

df = pd.DataFrame(trader_data)
df = df.sort_values("Error Frequency", ascending=False).reset_index(drop=True)


# Color only the Error Frequency column based on value
def highlight_error_frequency(val):
    if val > 0.2:
        return "background-color: #dc3545; color: white"  # Red with white text
    elif val > 0.1:
        return "background-color: #ffc107; color: black"  # Yellow with black text
    return ""


st.dataframe(
    df.style.applymap(highlight_error_frequency, subset=["Error Frequency"]).format(
        {
            "Error Frequency": "{:.1%}",
        }
    ),
    use_container_width=True,
    height=400,
)

# Download button
csv_data = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"reviewer_quality_{selected_project}.csv",
    mime="text/csv",
)

st.markdown("---")
