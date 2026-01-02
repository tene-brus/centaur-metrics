from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="View Results", page_icon="📊", layout="wide")
st.title("View Results")

# Help section explaining file categories
with st.expander("Understanding the Results", expanded=False):
    st.markdown("""
### File Categories

**overall_agreement** - Pairwise agreement scores between all annotators
- Shows how often each pair of annotators agreed on their annotations

**agreement_per_field** - Agreement broken down by annotation field
- `gt_breakdown_*` files show agreement per field type (e.g., direction, state_type)

**agreement_per_label** - Agreement broken down by label/category
- Shows how annotators performed on specific label values
- `gt_counts_*` files show counts where the annotator agreed with the ground truth

### Common vs Non-Common Tasks (concerns only rows where one of the annotators is ground_truth)
- `common_True` = Include tasks where the the annotator's submissions are marked as ground truth
- `common_False` = Do not include tasks where the the annotator's submissions are marked as ground truth

### Special Rows
- **primary_annotator = "ALL"**: Aggregated metrics for the entire trader (mean of all annotator pairs)
- **trader = "Total"**: Combined metrics across all traders (simple mean)

### Key Columns
- `trader`: The trading desk or team being evaluated
- `primary_annotator` / `secondary_annotator`: The pair of annotators being compared
- `prim_annot_tasks`: Number of tasks completed by the primary annotator
- `common_tasks`: Number of tasks where both annotators provided annotations
    """)

st.markdown("---")

# Paths - look in app/data for metrics directories
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Find all metrics/combined directories in app/data
metrics_dirs = [
    d
    for d in DATA_DIR.iterdir()
    if d.is_dir() and ("_metrics" in d.name or "combined" in d.name.lower())
]

if not metrics_dirs:
    st.warning("No metrics output directories found in app/data. Run metrics first.")
    st.stop()

# Select metrics directory
selected_dir = st.selectbox(
    "Select Metrics Directory",
    [d.name for d in sorted(metrics_dirs)],
    help="Choose a project's metrics folder or a combined output",
)

metrics_path = DATA_DIR / selected_dir

# Find only merged CSV files (merged_*.csv) in the selected directory, excluding flat directory
merged_files = [f for f in metrics_path.rglob("merged_*.csv") if "flat" not in f.parts]

if not merged_files:
    st.warning(f"No merged CSV files found in {selected_dir}. Run merge_csvs.py first.")
    st.stop()

# Group merged files by category (parent or grandparent directory)
categories: dict[str, list[Path]] = {}
for f in merged_files:
    # Get the category from parent directory or grandparent
    # e.g., overall_agreement/merged_overall_agreement.csv -> overall_agreement
    # or agreement_per_field/gt_breakdown_common_True/merged_*.csv -> agreement_per_field
    parent = f.parent.name
    if parent in ["agreement_per_field", "agreement_per_label", "overall_agreement"]:
        category = parent
    else:
        grandparent = f.parent.parent.name
        if grandparent in ["agreement_per_field", "agreement_per_label", "overall_agreement"]:
            category = grandparent
        else:
            continue  # Skip files that don't belong to a known category

    if category not in categories:
        categories[category] = []
    categories[category].append(f)

st.markdown("---")

# Category selection
col1, col2 = st.columns(2)

with col1:
    category_list = sorted(categories.keys())
    selected_category = st.selectbox(
        "Category",
        category_list,
        help="Type of analysis: overall_agreement, agreement_per_field, or agreement_per_label",
    )

with col2:
    # Get files for selected category
    files_in_category = categories.get(selected_category, [])

    # Create display names for files
    def get_file_display_name(path: Path) -> str:
        """Create a clean display name for a merged file."""
        filename = path.name
        # Remove 'merged_' prefix and '.csv' suffix
        clean_name = filename.replace("merged_", "").replace(".csv", "")
        return clean_name

    file_options = {get_file_display_name(f): f for f in sorted(files_in_category, key=get_file_display_name)}

    selected_file_name = st.selectbox(
        "File",
        list(file_options.keys()),
        help="Select the merged results file to view",
    )

file_path = file_options[selected_file_name]

st.markdown("---")

# Display selected file
st.subheader(f"{selected_category} / {selected_file_name}")

# Column descriptions for common columns
COLUMN_DESCRIPTIONS = {
    "trader": "The trading desk or team being evaluated",
    "primary_annotator": "First annotator in the pair ('ALL' = aggregated across all pairs)",
    "secondary_annotator": "Second annotator in the pair (None = aggregated)",
    "prim_annot_tasks": "Number of tasks completed by the primary annotator",
    "common_tasks": "Number of tasks where both annotators provided annotations",
}

try:
    df = pd.read_csv(file_path)
    original_row_count = len(df)

    # Column descriptions
    cols_with_desc = [c for c in df.columns if c in COLUMN_DESCRIPTIONS]
    if cols_with_desc:
        with st.expander("Column Descriptions", expanded=False, icon="📋"):
            for col in cols_with_desc:
                st.markdown(f"**{col}**: {COLUMN_DESCRIPTIONS[col]}")

    # Filtering section
    with st.expander("Filter Data", expanded=False, icon="🔍"):
        filtered_df = df.copy()

        # Get filterable columns (string/categorical columns)
        string_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object"
            or col
            in [
                "trader",
                "annotator",
                "primary_annotator",
                "secondary_annotator",
            ]
        ]

        # Get numeric columns for range filters
        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in ["float64", "int64", "float32", "int32"]
        ]

        # String/categorical filters
        if string_cols:
            st.markdown("**Categorical Filters**")
            filter_cols = st.columns(min(len(string_cols), 3))

            for i, col in enumerate(string_cols[:6]):  # Limit to 6 filters
                with filter_cols[i % 3]:
                    unique_vals = ["All"] + sorted(
                        df[col].dropna().unique().tolist()
                    )
                    selected_val = st.selectbox(
                        col,
                        unique_vals,
                        key=f"filter_{col}",
                    )
                    if selected_val != "All":
                        filtered_df = filtered_df[
                            filtered_df[col] == selected_val
                        ]

        # Numeric range filters
        if numeric_cols:
            st.markdown("**Numeric Filters**")

            # Select which numeric column to filter
            filter_numeric_col = st.selectbox(
                "Select numeric column to filter",
                ["None"] + numeric_cols,
                key="numeric_filter_col",
                help="Filter by agreement score or task count ranges",
            )

            if filter_numeric_col != "None":
                col_min = float(df[filter_numeric_col].min())
                col_max = float(df[filter_numeric_col].max())

                if col_min < col_max:
                    min_val, max_val = st.slider(
                        f"Range for {filter_numeric_col}",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        key=f"range_{filter_numeric_col}",
                    )
                    filtered_df = filtered_df[
                        (filtered_df[filter_numeric_col] >= min_val)
                        & (filtered_df[filter_numeric_col] <= max_val)
                    ]

        df = filtered_df

    # Show basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", original_row_count)
    with col2:
        st.metric("Filtered Rows", len(df))
    with col3:
        st.metric("Columns", len(df.columns))
    with col4:
        # Look for any agreement-like column
        agreement_cols = [
            c
            for c in df.columns
            if "agreement" in c.lower() or "mean" in c.lower()
        ]
        if agreement_cols and len(df) > 0:
            col_name = agreement_cols[0]
            if df[col_name].dtype in ["float64", "int64"]:
                st.metric(f"Mean {col_name}", f"{df[col_name].mean():.3f}")

    # Show dataframe
    st.dataframe(df, width="stretch", height=400)

    # Download button (filtered data)
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download CSV (filtered)"
        if len(df) < original_row_count
        else "Download CSV",
        data=csv_data,
        file_name=file_path.name,
        mime="text/csv",
    )

    # Show agreement heatmap for merged files with annotator columns
    if (
        "annotator_1" in df.columns
        and "annotator_2" in df.columns
        and any("agreement" in c.lower() for c in df.columns)
        and len(df) > 0
    ):
        st.markdown("---")
        st.subheader("Agreement Matrix Heatmap")

        # Find the agreement column
        agreement_col = next(c for c in df.columns if "agreement" in c.lower())

        # Pivot to matrix form
        pivot_df = df.pivot_table(
            index="annotator_1",
            columns="annotator_2",
            values=agreement_col,
            aggfunc="mean",
        )

        st.dataframe(
            pivot_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            width=True,
        )

except Exception as e:
    st.error(f"Error reading file: {e}")
