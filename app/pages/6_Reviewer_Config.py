"""Reviewer Configuration - Manage annotator exclusions per project."""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Reviewer Config", page_icon="👤", layout="wide")
st.title("Reviewer Configuration")
st.markdown("Manage which annotators (reviewers) should be excluded from metrics calculations.")
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
    return {
        "description": "Maps project names to reviewers that should be excluded from metrics.",
        "global_exclusions": [],
        "project_reviewers": {},
    }


def save_config(config: dict) -> None:
    """Save reviewer config to file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


# Load current config
config = load_config()

# Find available projects (JSONL files)
jsonl_files = list(DATA_DIR.glob("*.jsonl"))
project_names = [f.stem for f in jsonl_files]

# Also check for any projects in config that might not have JSONL files anymore
configured_projects = list(config.get("project_reviewers", {}).keys())
all_projects = sorted(set(project_names + configured_projects))

# ============================================================================
# Global Exclusions Section
# ============================================================================

st.subheader("Global Exclusions")
st.markdown("These annotators are excluded from **all** projects.")

global_exclusions = config.get("global_exclusions", [])

# Display current global exclusions
if global_exclusions:
    st.write("Currently excluded globally:")
    for email in global_exclusions:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(f"  • {email}")
        with col2:
            if st.button("Remove", key=f"remove_global_{email}"):
                global_exclusions.remove(email)
                config["global_exclusions"] = global_exclusions
                save_config(config)
                st.rerun()
else:
    st.info("No global exclusions configured.")

# Add new global exclusion
new_global = st.text_input("Add global exclusion (email)", key="new_global_email")
if st.button("Add Global Exclusion", key="add_global"):
    if new_global and new_global not in global_exclusions:
        global_exclusions.append(new_global)
        config["global_exclusions"] = global_exclusions
        save_config(config)
        st.success(f"Added {new_global} to global exclusions")
        st.rerun()
    elif new_global in global_exclusions:
        st.warning(f"{new_global} is already in global exclusions")

st.markdown("---")

# ============================================================================
# Project-Specific Reviewers Section
# ============================================================================

st.subheader("Project-Specific Reviewers")
st.markdown("These reviewers are excluded only from their specific project.")

project_reviewers = config.get("project_reviewers", {})

# Ensure all known projects have an entry
for project in all_projects:
    if project not in project_reviewers:
        project_reviewers[project] = []

if not all_projects:
    st.info("No projects found. Fetch some projects first.")
else:
    # Project selector
    selected_project = st.selectbox("Select Project", all_projects)

    if selected_project:
        reviewers = project_reviewers.get(selected_project, [])

        st.write(f"**Reviewers for {selected_project}:**")

        if reviewers:
            for email in reviewers:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"  • {email}")
                with col2:
                    if st.button("Remove", key=f"remove_{selected_project}_{email}"):
                        reviewers.remove(email)
                        project_reviewers[selected_project] = reviewers
                        config["project_reviewers"] = project_reviewers
                        save_config(config)
                        st.rerun()
        else:
            st.info(f"No project-specific reviewers for {selected_project}")

        # Add reviewer to project
        new_reviewer = st.text_input(
            f"Add reviewer to {selected_project} (email)",
            key=f"new_reviewer_{selected_project}",
        )
        if st.button("Add Project Reviewer", key=f"add_reviewer_{selected_project}"):
            if new_reviewer and new_reviewer not in reviewers:
                reviewers.append(new_reviewer)
                project_reviewers[selected_project] = reviewers
                config["project_reviewers"] = project_reviewers
                save_config(config)
                st.success(f"Added {new_reviewer} as reviewer for {selected_project}")
                st.rerun()
            elif new_reviewer in reviewers:
                st.warning(f"{new_reviewer} is already a reviewer for {selected_project}")

st.markdown("---")

# ============================================================================
# Summary Section
# ============================================================================

st.subheader("Configuration Summary")

with st.expander("View full configuration", expanded=False):
    st.json(config)

# Show effective exclusions per project
st.write("**Effective exclusions per project:**")
for project in all_projects:
    project_specific = project_reviewers.get(project, [])
    effective = list(set(global_exclusions + project_specific))
    if effective:
        st.write(f"• **{project}**: {', '.join(effective)}")
    else:
        st.write(f"• **{project}**: (none)")

st.markdown("---")

# Download/Upload config
st.subheader("Import/Export")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download Config",
        data=json.dumps(config, indent=2),
        file_name="reviewer_config.json",
        mime="application/json",
    )

with col2:
    uploaded_file = st.file_uploader("Upload Config", type=["json"])
    if uploaded_file is not None:
        try:
            uploaded_config = json.load(uploaded_file)
            if "global_exclusions" in uploaded_config or "project_reviewers" in uploaded_config:
                save_config(uploaded_config)
                st.success("Config uploaded successfully!")
                st.rerun()
            else:
                st.error("Invalid config format")
        except json.JSONDecodeError:
            st.error("Invalid JSON file")
