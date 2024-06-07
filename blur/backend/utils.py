"""Useful functions for experiments"""

import os
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm, trange

from blur.backend.config import SEED


def seed_everything(seed: int = SEED):
    """
    Seed further code

    :param seed: seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_unique(dataset: DatasetDict) -> dict:
    """
    Calculate unique values in dataset

    :param dataset: dataset with train, validation parts
    :return:
        Dict with unique values
    """
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


def invalid_filter(batch: Iterable) -> list[bool]:
    """
    Leave only invalid images

    :param batch: batch from dataset
    :return:
        List of bool values, True if images is invalid
    """
    results = []
    for item in batch:
        condition = all(face == 1 for face in item["invalid"])
        results.append(condition)

    return results


def custom_filter(batch: Iterable) -> list[bool]:
    """
    Leave only correct images

    :param batch: batch from dataset
    :return:
        List of bool values, True if images is invalid
    """
    results = []
    for item in batch:
        if (
            (len(item["bbox"]) < 3)
            and (len(item["bbox"]) > 0)
            and (all(face == 0 for face in item["invalid"]))
            and all(coo != 0 for bbox in item["bbox"] for coo in bbox)
        ):
            results.append(True)
        else:
            results.append(False)

    return results


def remove_people(dataset: DatasetDict, split: str) -> Dataset:
    """
    Process dataset to leave only valid images

    :param dataset: dataset dict to process
    :param split: split (train or validation)
    :return:
        Processed dataset
    """
    processed = dataset[split].filter(
        custom_filter,
        batched=True,
        batch_size=100,
        input_columns=["faces"],
        load_from_cache_file=False,
    )
    return processed


def prepare_df(dataset: Dataset) -> pd.DataFrame:
    """
    Prepare DataFrame with image info

    :param dataset: dataset to process
    :return:
        DataFrame with image information
    """
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
    """
    Prepare DatasetDict with image info

    :param dataset: DatasetDict to process
    :return:
        Dict with processed DataFrames
    """
    return {
        "train": prepare_df(dataset["train"]),
        "validation": prepare_df(dataset["validation"]),
    }
