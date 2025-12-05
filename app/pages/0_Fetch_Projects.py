import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Fetch Projects", page_icon="📥", layout="wide")
st.title("Fetch Projects from Label Studio")
st.markdown("---")

# Paths
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PROJECT_ROOT = APP_DIR.parent

# Check for .env file
env_file = PROJECT_ROOT / ".env"
if not env_file.exists():
    st.error(
        "No `.env` file found in project root. Please create one with LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY."
    )
    st.stop()

# Predefined projects (from update_projects.sh)
PREDEFINED_PROJECTS = [
    "trade extraction - signal1 - A",
    "trade extraction - signal1 - B",
]

st.subheader("Select Project")

# Option to use predefined or custom project name
project_source = st.radio("Project Source", ["Predefined", "Custom"])

if project_source == "Predefined":
    project_name = st.selectbox("Project", PREDEFINED_PROJECTS)
else:
    project_name = st.text_input(
        "Project Name", placeholder="Enter Label Studio project name"
    )

if not project_name:
    st.warning("Please enter a project name.")
    st.stop()

# Show output filename
output_filename = "_".join(project_name.replace("-", "").lower().split()) + ".jsonl"
output_path = DATA_DIR / output_filename
st.info(f"Output file: `{output_path}`")

st.markdown("---")


def run_fetch_with_progress(project_name: str, progress_bar, status_text):
    """Run fetch command and update progress bar in real-time."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "cli" / "get_project.py"),
        "--project_name",
        project_name,
        "--output_dir",
        str(DATA_DIR),
    ]

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Read stdout line by line for progress updates
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if line.startswith("PROGRESS:"):
            # Parse "PROGRESS:current/total"
            progress_str = line.replace("PROGRESS:", "")
            current, total = map(int, progress_str.split("/"))
            progress_bar.progress(current / total)
            status_text.text(f"Fetching task {current}/{total}")

    process.wait()
    return process.returncode, process.stderr.read()


# Fetch button
if st.button("Fetch Project", type="primary"):
    cmd_display = [
        sys.executable,
        str(PROJECT_ROOT / "cli" / "get_project.py"),
        "--project_name",
        project_name,
        "--output_dir",
        str(DATA_DIR),
    ]
    st.code(" ".join(cmd_display), language="bash")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Fetching project '{project_name}' from Label Studio...")

    try:
        returncode, stderr = run_fetch_with_progress(
            project_name, progress_bar, status_text
        )

        if returncode == 0:
            status_text.text("Complete!")
            st.success(f"Project fetched successfully! Saved to `{output_path}`")
        else:
            st.error("Error fetching project")
            if stderr:
                st.code(stderr, language="text")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# Fetch all predefined projects
st.subheader("Fetch All Predefined Projects")
st.markdown("Fetch all predefined projects at once.")

if st.button("Fetch All", type="secondary"):
    overall_progress = st.progress(0)
    overall_status = st.empty()

    for i, proj in enumerate(PREDEFINED_PROJECTS):
        overall_status.text(f"Fetching project {i + 1}/{len(PREDEFINED_PROJECTS)}: {proj}")

        st.write(f"**{proj}**")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            returncode, stderr = run_fetch_with_progress(proj, progress_bar, status_text)

            if returncode != 0:
                st.warning(f"Warning: {proj} had errors")
                if stderr:
                    st.code(stderr, language="text")
            else:
                status_text.text("Done!")
        except Exception as e:
            st.warning(f"Error fetching {proj}: {e}")

        overall_progress.progress((i + 1) / len(PREDEFINED_PROJECTS))

    overall_status.text("Complete!")
    st.success("All projects fetched!")

st.markdown("---")

# Show existing JSONL files in app/data
st.subheader("Existing Data Files")
jsonl_files = list(DATA_DIR.glob("*.jsonl"))

if jsonl_files:
    for f in sorted(jsonl_files):
        size_kb = f.stat().st_size / 1024
        st.text(f"  {f.name} ({size_kb:.1f} KB)")
else:
    st.text("No JSONL files found in app/data.")
