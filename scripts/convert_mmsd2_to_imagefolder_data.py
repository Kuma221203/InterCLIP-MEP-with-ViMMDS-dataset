import json
import shutil
import typing as t
from pathlib import Path

import datasets as dt
import jsonlines
from tqdm import tqdm

VERSION_TYPE = t.Literal["mmsd-v2"]

MMSD2_DATASET_DIR = Path("./data")

VERSION_NAME_MAP = {"mmsd-v2": "text_json_final"}

list_type_data = ["train", "validation", "test", "predict"]

def convert(version: VERSION_TYPE) -> None:
    data_dir = MMSD2_DATASET_DIR / VERSION_NAME_MAP[version]

    # create converted data dir
    converted_data_dir = MMSD2_DATASET_DIR / f"{version}-converted"
    converted_data_dir.mkdir(exist_ok=True, parents=True)
    flag = True

    for split in list_type_data:
        split_dir = converted_data_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)

        metadata_file = (split_dir / "metadata.jsonl").open("w")
        metadata_writer = jsonlines.Writer(metadata_file)

        data = json.loads((data_dir / f"{split}.json").read_text())

        for d in tqdm(data, desc=f"Converting {version} {split} data"):
            image_id = d["image_id"]
            text = d["text"]
            label = d["label"]
            image_path = MMSD2_DATASET_DIR / "dataset_image" / f"{image_id}.jpg"
            if not image_path.exists():
                continue

            metadata_writer.write(
                {
                    "file_name": f"{image_id}.jpg",
                    "text": text,
                    "label": -1 if split == "predict" else label,
                    "id": str(image_id),
                }
            )
                
            shutil.copy(image_path, split_dir / f"{image_id}.jpg")

def publish(version: VERSION_TYPE, repo_id: str) -> None:
    converted_data_dir = MMSD2_DATASET_DIR / f"{version}-converted"
    dataset_dict = {}

    for split in list_type_data:
        split_dir = converted_data_dir / split
        if split_dir.exists():  
            dataset_split = dt.load_dataset("imagefolder", data_dir=str(split_dir), split="train")
            dataset_dict[split] = dataset_split  

    dataset = dt.DatasetDict(dataset_dict)
    dataset.push_to_hub(repo_id, config_name=version)

if __name__ == "__main__":
    convert("mmsd-v2")
    publish("mmsd-v2", "kuma22/MMSD2.0")

