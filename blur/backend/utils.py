import os
import random
from tqdm import tqdm
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict
from blur.backend.config import SEED


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
        condition = (
            all(face == 1 for face in item["invalid"])
        )
        results.append(condition)
        
    return results
    

def custom_filter(batch) -> list[bool]:
    results = []
    for item in batch:
        if (
            (len(item["bbox"]) < 3) and (len(item["bbox"]) > 0) and 
            (all(face == 0 for face in item["invalid"]))
        ):
            results.append(True)
        else:
            results.append(False)
            
    return results


def remove_people(dataset: DatasetDict, split: str):
    processed = dataset[split].filter(
        custom_filter,
        batched=True, batch_size=100,
        input_columns=["faces"],
    ) 
    return processed