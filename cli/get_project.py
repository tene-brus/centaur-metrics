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

client = LabelStudio(
    base_url=os.getenv("LABEL_STUDIO_URL"), api_key=os.getenv("LABEL_STUDIO_API_KEY")
)


def get_project_metadata(client: LabelStudio, project_name: str):
    response = client.projects.list()

    for item in response:
        if item.title == project_name:
            break

    return item.id, item.members, item.task_number


def extract_annotations(annotation_result: dict, specific_field: str | None = None):
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

            annotations[annot["id"]][field] = annot["value"][value_key][0]

    if specific_field is not None:
        return [value[specific_field] for _, value in annotations.items()]
    else:
        return [value for _, value in annotations.items()]


def get_project_annotations(
    client: LabelStudio,
    project_name: str | None = None,
    specific_field: bool = False,
    field: str | None = None,
):
    if specific_field and not field:
        raise RuntimeError("If 'specific_field' is True, then field should be a string")

    project_id, members, num_tasks = get_project_metadata(
        client=client, project_name=project_name
    )

    logger.info(f"Project: {project_name}")

    logger.info(f"Amount of members: {len(members)}")

    logger.info(f"Amount of tasks: {num_tasks}")

    tasks = client.tasks.list(project=project_id)

    file_name = "_".join(project_name.replace("-", "").lower().split())

    if field:
        logger.info(f"Extracting on one file per annotation: {field}")
        file_name += f"_{field}"

    with open(f"{file_name}.jsonl", "w", encoding="utf-8") as file:
        for task in tqdm(
            tasks, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", total=num_tasks
        ):
            entry = {annotator["user"]["email"].strip(): None for annotator in members}
            entry["predictions"] = None
            entry["ground_truth_member"] = None
            entry["ground_truth"] = None
            entry["id"] = task.id
            entry["num_annotations"] = len(task.annotations)
            for i in range(task.annotators_count):
                try:
                    annotator = task.annotations[i]["completed_by"]["email"].strip()
                    annotations = extract_annotations(
                        task.annotations[i]["result"], specific_field=field
                    )
                    entry[annotator] = annotations

                    if task.annotations[i]["ground_truth"]:
                        entry["ground_truth"] = annotations
                        entry["ground_truth_member"] = task.annotations[i][
                            "completed_by"
                        ]["email"].strip()

                    predictions = extract_annotations(
                        task.predictions[0].result, specific_field=field
                    )
                    entry["predictions"] = predictions

                except Exception as e:
                    logging.error(f"{str(e)}")

            entry = json.dumps(entry)

            file.write(entry + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--field", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.field:
        specific_field = True
    else:
        specific_field = False

    get_project_annotations(
        client,
        project_name=args.project_name,
        specific_field=specific_field,
        field=args.field,
    )
