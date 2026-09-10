#!/bin/bash
# Fetch the latest annotation data for one or more Label Studio projects.
#
# Usage: ./update_projects.sh "project name" ["another project" ...]

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: ./update_projects.sh \"project name\" [\"another project\" ...]"
    echo "Example: ./update_projects.sh \"trade extraction - signal1 - A\""
    exit 1
fi

source .venv/bin/activate

for project_name in "$@"; do
    echo "Fetching: $project_name"
    python cli/get_project.py --project_name "$project_name"
done
