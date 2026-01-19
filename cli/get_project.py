import argparse
import json
import logging
import os

from dotenv import load_dotenv
from label_studio_sdk import LabelStudio
from tqdm import tqdm

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

client = LabelStudio(
    base_url=os.getenv("LABEL_STUDIO_URL"), api_key=os.getenv("LABEL_STUDIO_API_KEY")
)

OPTIONAL_STATE_FLAGS = [
    "state_optional_task_flags",
    "action_optional_task_flags",
]

# GT Verifier user IDs -> emails
GT_VERIFIER_IDS = {
    73379: "alissa@zap.xyz",
    73374: "tony@zap.xyz",
}


def get_project_metadata(client: LabelStudio, project_name: str):
    response = client.projects.list()

    for item in response:
        if item.title == project_name:
            break

    return item.id, item.members, item.task_number


def extract_annotations(annotation_result: dict):
    ids = set()
    for annot in annotation_result:
        ids.add(annot["id"])

    annotations = {idx: {} for idx in ids}

    for annot in annotation_result:
        # for label in annot:
        field = annot["from_name"]
        if annot["type"] == "taxonomy":
            value_key = "taxonomy"

            annotations[annot["id"]][field] = annot["value"][value_key][0]

        elif annot["type"] == "labels":
            value_key = "labels"

            annotations[annot["id"]][field] = annot["value"][value_key][0]

        elif annot["type"] == "choices":
            value_key = "choices"

            if field not in OPTIONAL_STATE_FLAGS:
                annotations[annot["id"]][field] = annot["value"][value_key][0]
            else:
                annotations[annot["id"]][field] = annot["value"][value_key]

    return [value for _, value in annotations.items()]


def get_project_annotations(
    client: LabelStudio,
    project_name: str | None = None,
    output_dir: str | None = None,
):
    project_id, members, num_tasks = get_project_metadata(
        client=client, project_name=project_name
    )

    logger.info(f"Project: {project_name}")

    logger.info(f"Amount of members: {len(members)}")

    logger.info(f"Amount of tasks: {num_tasks}")

    tasks = client.tasks.list(project=project_id)

    file_name = "_".join(project_name.replace("-", "").lower().split())

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{file_name}.jsonl")
    else:
        file_path = f"{file_name}.jsonl"

    with open(file_path, "w", encoding="utf-8") as file:
        for idx, task in enumerate(
            tqdm(tasks, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", total=num_tasks)
        ):
            # Print progress for external tools (e.g., Streamlit)
            if idx % 100 == 0:
                print(f"PROGRESS:{idx + 1}/{num_tasks}", flush=True)

            entry = {annotator["user"]["email"].strip(): None for annotator in members}
            entry["predictions"] = None
            entry["ground_truth_member"] = None
            entry["ground_truth"] = None
            entry["gt_accepted"] = False
            entry["gt_accepted_by"] = None
            entry["id"] = task.id
            entry["num_annotations"] = len(task.annotations)
            entry["trader"] = task.data["Trader"]

            gt_annotation_id = None
            for i in range(task.annotators_count):
                try:
                    annotator = task.annotations[i]["completed_by"]["email"].strip()
                    annotations = extract_annotations(task.annotations[i]["result"])
                    entry[annotator] = annotations

                    if task.annotations[i]["ground_truth"]:
                        entry["ground_truth"] = annotations
                        entry["ground_truth_member"] = task.annotations[i][
                            "completed_by"
                        ]["email"].strip()
                        gt_annotation_id = task.annotations[i]["id"]

                    predictions = extract_annotations(task.predictions[0].result)
                    entry["predictions"] = predictions

                except Exception as e:
                    logging.error(f"{str(e)}")

            # Fetch who accepted the GT annotation (only if GT exists)
            if gt_annotation_id is not None:
                try:
                    reviews = client.annotation_reviews.list(
                        annotation=gt_annotation_id
                    )
                    for review in reviews:
                        if review.accepted:
                            entry["gt_accepted"] = True
                            if review.created_by in GT_VERIFIER_IDS:
                                entry["gt_accepted_by"] = GT_VERIFIER_IDS[
                                    review.created_by
                                ]
                            break
                except Exception as review_err:
                    logger.warning(
                        f"Could not fetch reviews for annotation {gt_annotation_id}: {review_err}"
                    )

            entry = json.dumps(entry)

            file.write(entry + "\n")

    print(f"PROGRESS:{num_tasks}/{num_tasks}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default=None)

    args = parser.parse_args()

    get_project_annotations(
        client,
        project_name=args.project_name,
        output_dir=args.output_dir,
    )
