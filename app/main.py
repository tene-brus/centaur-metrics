import streamlit as st

st.set_page_config(
    page_title="Agreement Metrics",
    page_icon="📊",
    layout="wide",
)

st.title("Agreement Metrics Dashboard")
st.markdown("---")

st.markdown("""
## Welcome

This dashboard allows you to calculate and explore inter-annotator agreement metrics.

### Pages

1. **Fetch Projects** - Download annotation data from Label Studio with real-time progress tracking
2. **Run Metrics** - Calculate agreement metrics (overall, per-field, per-label) with optional per-trader breakdowns
3. **Merge CSVs** - Combine per-trader CSV files into merged summaries
4. **Combine Projects** - Merge metrics from two different projects for comparison
5. **View Results** - Browse, filter, and download the generated CSV files with heatmap visualizations

### Getting Started

1. Use **Fetch Projects** to download your Label Studio project data
2. Run **Run Metrics** (or "Run All" for comprehensive analysis)
3. Optionally **Merge CSVs** to consolidate per-trader files
4. Explore results in **View Results** with filtering and export options

Use the sidebar to navigate between pages.
""")

st.sidebar.success("Select a page above.")
