import os
from functools import partial
from typing import TypedDict, cast

from underthesea import word_tokenize
import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer


class MMSDModelInput(TypedDict, total=False):
    pixel_values: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    label: Tensor
    id: list[str]

def preprocess(example, image_processor, tokenizer):
    image_inputs = image_processor(
        images=example["image"],
    )
    tokenized_texts = [word_tokenize(text, format="text") for text in example["text"]]
    text_inputs = tokenizer(
        text=tokenized_texts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    return {
        "pixel_values": image_inputs["pixel_values"],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "label": example["label"],
        "id": example["id"],
    }


class MMSDDatasetModule(pl.LightningDataModule):

    def __init__(
        self,
        vision_ckpt_name: str,
        text_ckpt_name: str,
        dataset_version: str = "vimmsd-v2",
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.vision_ckpt_name = vision_ckpt_name
        self.text_ckpt_name = text_ckpt_name

        self.dataset_version = dataset_version
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        image_processor = AutoImageProcessor.from_pretrained(self.vision_ckpt_name)
        tokenizer = AutoTokenizer.from_pretrained(self.text_ckpt_name)

        self.dataset = cast(
            Dataset,
            load_dataset("kuma22/VIMMSD2.0", name=self.dataset_version),
        )
        self.dataset.set_transform(
            partial(
                preprocess,
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],  # type: ignore
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["test"],  # type: ignore
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["predict"],
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, batch) -> MMSDModelInput:
        return {
            "pixel_values": torch.stack(
                [torch.tensor(x["pixel_values"]) for x in batch]
            ),
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack(
                [torch.tensor(x["attention_mask"]) for x in batch]
            ),
            "label": torch.stack([torch.tensor(x["label"]) for x in batch]),
            "id": [x["id"] for x in batch],
        }
