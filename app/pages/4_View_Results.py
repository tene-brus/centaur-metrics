from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="View Results", page_icon="📊", layout="wide")
st.title("View Results")
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
    "Select Metrics Directory", [d.name for d in sorted(metrics_dirs)]
)

metrics_path = DATA_DIR / selected_dir

# Find all CSV files in the selected directory
csv_files = list(metrics_path.rglob("*.csv"))

if not csv_files:
    st.warning(f"No CSV files found in {selected_dir}")
    st.stop()


# Group files by category
def categorize_file(path: Path) -> tuple[str, str, str]:
    parts = path.relative_to(metrics_path).parts
    if len(parts) >= 3:
        # e.g., overall_agreement/common_False/file.csv
        category = parts[0]
        subcategory = parts[1]
        return category, subcategory, path.name
    elif len(parts) == 2:
        # e.g., flat/file.csv or category/file.csv
        category = parts[0]
        return category, "(root)", path.name
    else:
        # Files directly in metrics_path
        return "(root)", "(root)", path.name


# Build file tree
file_tree: dict[str, dict[str, list[Path]]] = {}
for f in csv_files:
    cat, subcat, _ = categorize_file(f)
    if cat not in file_tree:
        file_tree[cat] = {}
    if subcat not in file_tree[cat]:
        file_tree[cat][subcat] = []
    file_tree[cat][subcat].append(f)

st.markdown("---")

# Category selection
col1, col2, col3 = st.columns(3)

with col1:
    categories = sorted(file_tree.keys())
    selected_category = st.selectbox("Category", categories)

with col2:
    subcategories = sorted(file_tree.get(selected_category, {}).keys())
    selected_subcategory = st.selectbox("Subcategory", subcategories)

with col3:
    files_in_subcat = file_tree.get(selected_category, {}).get(selected_subcategory, [])
    file_names = sorted([f.name for f in files_in_subcat])
    selected_file = st.selectbox("File", file_names)

st.markdown("---")

# Display selected file
if selected_file:
    file_path = next((f for f in files_in_subcat if f.name == selected_file), None)

    if file_path:
        # Show breadcrumb
        if selected_subcategory == "(root)":
            breadcrumb = f"{selected_category} / {selected_file}"
        else:
            breadcrumb = (
                f"{selected_category} / {selected_subcategory} / {selected_file}"
            )
        st.subheader(breadcrumb)

        try:
            df = pd.read_csv(file_path)
            original_row_count = len(df)

            # Filtering section
            with st.expander("Filter Data", expanded=False):
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

                # Text search
                st.markdown("**Text Search**")
                search_text = st.text_input("Search in all columns", key="search_text")
                if search_text:
                    mask = (
                        filtered_df.astype(str)
                        .apply(
                            lambda x: x.str.contains(search_text, case=False, na=False)
                        )
                        .any(axis=1)
                    )
                    filtered_df = filtered_df[mask]

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
            st.dataframe(df, width=True, height=400)

            # Download button (filtered data)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV (filtered)"
                if len(df) < original_row_count
                else "Download CSV",
                data=csv_data,
                file_name=selected_file,
                mime="text/csv",
            )

            # Show agreement heatmap for Total_agreement or merged files with annotator columns
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
