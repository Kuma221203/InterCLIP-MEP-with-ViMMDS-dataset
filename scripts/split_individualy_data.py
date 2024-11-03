import json
import shutil
import typing as t
from pathlib import Path

import datasets as dt
import jsonlines
from tqdm import tqdm

VERSION_TYPE = t.Literal["vimmsd"]

VIMMSD_DATASET_DIR = Path("./data")

VERSION_NAME_MAP = {"vimmsd": "text_json"}


def process_split(version: VERSION_TYPE, split_name: str, image_folder: str) -> None:
    data_dir = VIMMSD_DATASET_DIR / VERSION_NAME_MAP[version]

    # create converted data dir for the specific split
    converted_data_dir = VIMMSD_DATASET_DIR / f"{version}-converted"
    split_dir = converted_data_dir / split_name
    split_dir.mkdir(exist_ok=True, parents=True)

    # create metadata file
    metadata_file = (split_dir / "metadata.jsonl").open("w")
    metadata_writer = jsonlines.Writer(metadata_file)

    # copy images and write metadata
    data = json.loads((data_dir / f"{split_name}.json").read_text())

    for d in tqdm(data, desc=f"Converting {version} {split_name} data"):
        image_id = d["image_id"]
        text = d["text"]
        label = d.get("label", None)  # Only use label if present
        image_path = VIMMSD_DATASET_DIR / f"{image_folder}" / f"{image_id}.jpg"
        if not image_path.exists():
            continue

        metadata_writer.write(
            {
                "file_name": f"{image_id}.jpg",
                "text": text,
                "label": -1 if split_name == "predict" else label,
                "id": str(image_id),
            }
        )
        shutil.copy(image_path, split_dir / f"{image_id}.jpg")


def publish(version: VERSION_TYPE, repo_id: str, split_name: str) -> None:
    """Publish a single split to the Hugging Face Hub"""
    converted_data_dir = VIMMSD_DATASET_DIR / f"{version}-converted" / split_name

    if converted_data_dir.exists():
        dataset_split = dt.load_dataset("imagefolder", data_dir=str(converted_data_dir), split="train")
        dataset_split.push_to_hub(repo_id, config_name=f"{version}_{split_name}")
    else:
        print(f"Error: {split_name} data does not exist.")

if __name__ == "__main__":
    process_split("vimmsd", 'vimmsd-warmup', "warmup-images")
    publish("vimmsd", "kuma22/vimmsd")
