"""Ground Truth Quality Dashboard - View annotator agreement with GT by field/label."""

import json
from pathlib import Path

import polars as pl
import streamlit as st

# Note: Ambiguous labels like "Unclear" are now output as field-specific columns
# (e.g., "Unclear (direction)", "Unclear (exposure_change)") directly from the metrics pipeline


st.set_page_config(page_title="GT Quality", page_icon="🎯", layout="wide")
st.title("Ground Truth Quality Dashboard")
st.markdown(
    "View annotator agreement with Ground Truth, filterable by trader and field/label."
)
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


# Find metrics directories (both *_metrics and combined*)
metrics_dirs = [
    d
    for d in DATA_DIR.iterdir()
    if d.is_dir() and ("_metrics" in d.name or "combined" in d.name.lower())
]

if not metrics_dirs:
    st.warning("No metrics directories found. Run metrics first.")
    st.stop()

# Select project
selected_project = st.selectbox(
    "Select Project",
    [d.name for d in sorted(metrics_dirs)],
    help="Choose which project's metrics to view",
)

metrics_path = DATA_DIR / selected_project

# Check for GT breakdown files - handle both regular and combined directory structures
# Regular: agreement_per_field/gt_breakdown_common_False/*.csv
# Combined: agreement_per_field/merged_gt_breakdown_common_False.csv

gt_field_path = metrics_path / "agreement_per_field" / "gt_breakdown_common_True"
gt_field_path_combined = metrics_path / "agreement_per_field"
gt_label_path = metrics_path / "agreement_per_label" / "common_True"
gt_label_path_combined = metrics_path / "agreement_per_label"

# Check for gt_counts paths
gt_counts_path = metrics_path / "agreement_per_label" / "gt_counts_common_True"
gt_counts_path_combined = metrics_path / "agreement_per_label"

# Check which paths have data
has_field_data = (gt_field_path.exists() and list(gt_field_path.glob("*.csv"))) or (
    gt_field_path_combined.exists()
    and list(gt_field_path_combined.glob("*gt_breakdown*.csv"))
)
has_label_data = (gt_label_path.exists() and list(gt_label_path.glob("*.csv"))) or (
    gt_label_path_combined.exists()
    and list(gt_label_path_combined.glob("merged_common*.csv"))
)
has_counts_data = (gt_counts_path.exists() and list(gt_counts_path.glob("*.csv"))) or (
    gt_counts_path_combined.exists()
    and list(gt_counts_path_combined.glob("*gt_counts*.csv"))
)

if not has_field_data and not has_label_data and not has_counts_data:
    st.warning(
        "No GT breakdown data found. Run metrics with per-field or per-label case."
    )
    st.stop()

st.markdown("---")

# View type selection
available_views = []
if has_field_data:
    available_views.append("Per Field")
if has_label_data:
    available_views.append("Per Label")
if has_counts_data:
    available_views.append("Per Label (Counts)")

view_type = st.radio(
    "View Type",
    available_views,
    horizontal=True,
    help="Per Field: agreement on direction, state_type, etc. | Per Label: agreement on specific label values | Per Label (Counts): raw count of agreements with GT",
)

if view_type == "Per Field" and not has_field_data:
    st.warning("Per-field GT breakdown data not available for this project.")
    st.stop()
elif view_type == "Per Label" and not has_label_data:
    st.warning("Per-label GT breakdown data not available for this project.")
elif view_type == "Per Label (Counts)" and not has_counts_data:
    st.warning("Per-label counts data not available for this project.")
    st.stop()


def load_gt_data(base_path: Path, combined_path: Path, pattern: str) -> pl.DataFrame:
    """Load GT comparison data from merged CSV file only.

    Only loads merged files to avoid double-counting from individual trader files.
    """
    merged_file = None

    # Check for merged file in base_path
    if base_path.exists():
        merged_files = list(base_path.glob("merged_*.csv"))
        if merged_files:
            merged_file = merged_files[0]

    # If no merged file in base_path, check combined structure
    if not merged_file and combined_path.exists():
        merged_files = list(combined_path.glob(pattern))
        if merged_files:
            merged_file = merged_files[0]

    if not merged_file:
        return pl.DataFrame()

    try:
        df = pl.read_csv(merged_file)
        # Filter to only GT comparisons
        if "secondary_annotator" in df.columns:
            df = df.filter(pl.col("secondary_annotator") == "ground_truth")
        # Exclude rows where common_tasks is 0 (no GT comparisons)
        if "common_tasks" in df.columns:
            df = df.filter(pl.col("common_tasks") > 0)
        return df
    except Exception:
        return pl.DataFrame()


# Load data based on view type
is_counts_view = view_type == "Per Label (Counts)"

if view_type == "Per Field":
    df = load_gt_data(
        gt_field_path, gt_field_path_combined, "*gt_breakdown_common_True*.csv"
    )
    # Fields to show (excluding metadata columns)
    field_columns = [
        "state_type",
        "direction",
        "exposure_change",
        "position_status",
        "optional_task_flags",
    ]
    field_columns = [c for c in field_columns if c in df.columns]
elif view_type == "Per Label":
    df = load_gt_data(gt_label_path, gt_label_path_combined, "merged_common_True*.csv")
    # For per-label, identify label columns (all non-metadata columns)
    metadata_cols = [
        "primary_annotator",
        "secondary_annotator",
        "prim_annot_tasks",
        "common_tasks",
        "trader",
        "sum_contrib",
    ]
    field_columns = [c for c in df.columns if c not in metadata_cols]
else:  # Per Label (Counts)
    df = load_gt_data(
        gt_counts_path, gt_counts_path_combined, "*gt_counts_common_True*.csv"
    )
    # For per-label counts, identify label columns (all non-metadata columns)
    metadata_cols = [
        "primary_annotator",
        "secondary_annotator",
        "prim_annot_tasks",
        "common_tasks",
        "trader",
        "sum_contrib",
    ]
    field_columns = [c for c in df.columns if c not in metadata_cols]

if df.is_empty():
    st.warning(
        "No merged data file found. Run the merge script first to generate merged CSV files."
    )
    st.stop()

# Exclude "Total" rows from everywhere
df = df.filter(pl.col("trader") != "Total")

if df.is_empty():
    st.warning("No data available after excluding Total rows.")
    st.stop()

# Filters
st.subheader("Filters")
col1, col2 = st.columns(2)

with col1:
    # Trader filter
    traders = ["All"] + sorted(
        df.select("trader").drop_nulls().unique().to_series().to_list()
    )
    selected_trader = st.selectbox(
        "Trader", traders, help="Filter by specific trader or view all"
    )

with col2:
    # Field/Label filter
    if view_type == "Per Field":
        selected_fields = st.multiselect(
            "Fields",
            field_columns,
            default=field_columns,
            help="Select which fields to display",
        )
    else:
        # Labels are already disambiguated in column names (e.g., "Unclear (direction)")
        default_labels = (
            field_columns[:10] if len(field_columns) > 10 else field_columns
        )
        selected_fields = st.multiselect(
            "Labels",
            field_columns,
            default=default_labels,
            help="Select which labels to display",
        )

if not selected_fields:
    st.warning("Please select at least one field/label.")
    st.stop()

# Apply trader filter
if selected_trader != "All":
    df = df.filter(pl.col("trader") == selected_trader)

if df.is_empty():
    st.warning("No data matches the selected filters.")
    st.stop()

st.markdown("---")

# Display results
if is_counts_view:
    st.subheader("Annotator Agreement Counts with Ground Truth")
else:
    st.subheader("Annotator Agreement with Ground Truth")

# Prepare display dataframe
display_cols = ["primary_annotator", "trader", "common_tasks"] + selected_fields
display_df = df.select(display_cols)

# Convert common_tasks to int
display_df = display_df.with_columns(pl.col("common_tasks").cast(pl.Int64))

# For counts view, convert selected fields to int as well
if is_counts_view:
    display_df = display_df.with_columns(
        [
            pl.col(col).cast(pl.Int64)
            for col in selected_fields
            if col in display_df.columns
        ]
    )

# Rename metadata columns for clarity (label columns are already properly named)
rename_cols = {"primary_annotator": "Annotator", "common_tasks": "Tasks vs GT"}
display_df = display_df.rename(rename_cols)

# Display field names are the same as selected fields (already disambiguated)
display_field_names = selected_fields

# Sort by first selected field descending
if display_field_names:
    display_df = display_df.sort(display_field_names[0], descending=True)

# Show metrics summary (use original field names for data access, display names for labels)
st.markdown("**Summary Statistics (Weighted by Task Count)**")
summary_cols = st.columns(len(selected_fields[:5]))  # Limit to 5 in summary row
for i, field in enumerate(selected_fields[:5]):
    display_name = display_field_names[i] if i < len(display_field_names) else field
    with summary_cols[i]:
        if is_counts_view:
            total_val = df.select(pl.col(field).sum()).item()
            st.metric(display_name, f"{int(total_val):,}")
        else:
            # Weighted mean: sum(score * tasks) / sum(tasks)
            weighted_sum = df.select(
                (pl.col(field) * pl.col("common_tasks")).sum()
            ).item()
            total_tasks = df.select(pl.col("common_tasks").sum()).item()
            weighted_mean = weighted_sum / total_tasks if total_tasks > 0 else 0
            st.metric(display_name, f"{weighted_mean:.3f}")

st.markdown("---")

# Convert to pandas for streamlit display with styling
display_df_pandas = display_df.to_pandas()

# Show the data table with appropriate formatting
if is_counts_view:
    # For counts, no gradient (values aren't 0-1), format as integers
    st.dataframe(
        display_df_pandas.style.format({col: "{:,.0f}" for col in display_field_names}),
        use_container_width=True,
        height=400,
    )
else:
    # For ratios, use gradient and 3 decimal formatting
    st.dataframe(
        display_df_pandas.style.background_gradient(
            subset=display_field_names,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        ).format({col: "{:.3f}" for col in display_field_names}),
        use_container_width=True,
        height=400,
    )

# Aggregated view - mean/sum per annotator across all traders
if selected_trader == "All":
    st.markdown("---")

    if is_counts_view:
        st.subheader("Aggregated by Annotator (Sum Across All Traders)")
        agg_df = df.group_by("primary_annotator").agg(
            [pl.col("common_tasks").sum()]
            + [pl.col(field).sum() for field in selected_fields]
        )
    else:
        st.subheader("Aggregated by Annotator (Weighted Mean Across All Traders)")
        # Weighted mean: sum(score * tasks) / sum(tasks) for each field
        # This matches the View Results calculation
        agg_df = (
            df.group_by("primary_annotator")
            .agg(
                [pl.col("common_tasks").sum()]
                + [
                    (pl.col(field) * pl.col("common_tasks"))
                    .sum()
                    .alias(f"{field}_weighted")
                    for field in selected_fields
                ]
            )
            # Divide weighted sums by total tasks to get weighted mean
            .with_columns(
                [
                    (pl.col(f"{field}_weighted") / pl.col("common_tasks")).alias(field)
                    for field in selected_fields
                ]
            )
            # Drop the intermediate weighted columns
            .drop([f"{field}_weighted" for field in selected_fields])
            # Make total contribution column (mean of the weighted means)
            .with_columns(
                pl.mean_horizontal(
                    pl.exclude("primary_annotator", "common_tasks")
                ).alias("average")
            )
        )
        # Add it to the display list & sort based on it
        display_field_names += ["average"]
        agg_df = agg_df.sort("average", descending=True)

    # Rename metadata columns (label columns are already properly named)
    agg_rename_cols = {
        "primary_annotator": "Annotator",
        "common_tasks": "Total Tasks vs GT",
    }
    agg_df = agg_df.rename(agg_rename_cols)
    agg_df = agg_df.with_columns(pl.col("Total Tasks vs GT").cast(pl.Int64))

    # Convert to pandas for styling
    agg_df_pandas = agg_df.to_pandas()

    if is_counts_view:
        # Convert to int for counts
        for col in display_field_names:
            if col in agg_df_pandas.columns:
                agg_df_pandas[col] = agg_df_pandas[col].astype(int)
        st.dataframe(
            agg_df_pandas.style.format({col: "{:,.0f}" for col in display_field_names}),
            use_container_width=True,
        )
    else:
        st.dataframe(
            agg_df_pandas.style.background_gradient(
                subset=display_field_names,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
            ).format({col: "{:.3f}" for col in display_field_names}),
            use_container_width=True,
        )

# Download button
csv_data = display_df.write_csv()
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"gt_quality_{selected_project}_{selected_trader}.csv",
    mime="text/csv",
)
