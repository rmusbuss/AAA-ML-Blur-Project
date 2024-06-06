"""Useful functions for experiments"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from blur.backend.config import SEED
from datasets import Dataset, DatasetDict
from PIL import ImageDraw
from tqdm import tqdm, trange


def seed_everything(seed: int = SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_unique(dataset: DatasetDict) -> dict:
    unique = {}
    for split in ["train", "validation"]:
        unique[split] = {
            "blur": {0: 0, 1: 0, 2: 0},
            "expression": {0: 0, 1: 0},
            "illumination": {0: 0, 1: 0},
            "occlusion": {0: 0, 1: 0, 2: 0},
            "pose": {0: 0, 1: 0},
            "invalid": {0: 0, 1: 0},
        }
        for img in tqdm(dataset[split]["faces"]):
            for iface in range(len(img["bbox"])):
                for feature in unique[split].keys():
                    value = int(img[feature][iface])
                    unique[split][feature][value] += 1
    return unique


def invalid_filter(batch) -> list[bool]:
    results = []
    for item in batch:
        condition = all(face == 1 for face in item["invalid"])
        results.append(condition)

    return results


def custom_filter(batch) -> list[bool]:
    results = []
    for item in batch:
        if (
            (len(item["bbox"]) < 3)
            and (len(item["bbox"]) > 0)
            and (all(face == 0 for face in item["invalid"]))
            and (all(all([coo != 0 for coo in bbox]) for bbox in item["bbox"]))
        ):
            results.append(True)
        else:
            results.append(False)

    return results


def remove_people(dataset: DatasetDict, split: str):
    processed = dataset[split].filter(
        custom_filter,
        batched=True,
        batch_size=100,
        input_columns=["faces"],
        load_from_cache_file=False,
    )
    return processed


def prepare_df(dataset: Dataset) -> pd.DataFrame:
    features = [
        "blur",
        "expression",
        "illumination",
        "occlusion",
        "pose",
        "invalid",
    ]
    sizes = []
    for idx in trange(len(dataset)):
        sizes.append(dataset[idx]["image"].size)

    df = pd.DataFrame(dataset["faces"])
    df[["size_w", "size_h"]] = sizes

    df = (
        df.explode(column=["bbox", *features])
        .reset_index()
        .rename(columns={"index": "img_id"})
    )

    df[["xmin", "ymin", "width", "height"]] = df["bbox"].tolist()
    df["xmax"] = df["xmin"] + df["width"]
    df["ymax"] = df["ymin"] + df["height"]

    df = df.drop(columns=["bbox"])
    coords = [
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "width",
        "height",
        "size_w",
        "size_h",
    ]
    df["label"] = "face"
    df = df[["img_id", "label", *coords, *features]]
    return df


def prepare_dfs(dataset: DatasetDict) -> dict[str, pd.DataFrame]:
    return {
        "train": prepare_df(dataset["train"]),
        "validation": prepare_df(dataset["validation"]),
    }
