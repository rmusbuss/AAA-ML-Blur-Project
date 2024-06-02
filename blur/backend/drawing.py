import os
import random
from tqdm import tqdm
import pandas as pd
from PIL import ImageDraw
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from datasets import DatasetDict


def draw_faces_with_bbox(
    dataset: DatasetDict, split: str = 'train', 
    n_rows: int = 2, n_cols: int = 4,
    hide_face: bool = True,
):
    if split not in ["train", "validation", "test"]:
        raise ValueError("Unexpected split.")
        
    data = dataset[split]
    num_images = n_rows * n_cols
    indices = random.sample(range(data.num_rows), num_images)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    plt.suptitle(f"{split} images", fontsize=16)
    axes = axes.flatten()
    
    for ax, idx in zip(axes, indices):
        example = data[idx]
        ax.set_title(idx)
        image = example['image']
        draw = ImageDraw.Draw(image)
        for bbox in example['faces']["bbox"]:
            x_min, y_min, width, height = bbox
            x_max, y_max = x_min + width, y_min + height
            bbox = [x_min, y_min, x_max, y_max]
            
            if hide_face:
                draw.rectangle(bbox, fill="black")
            else:
                draw.rectangle(bbox, outline="red", width=15)
                
        ax.imshow(image)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_categories(data: dict, suptitle: str):
    colors = ['skyblue', 'orange', 'lightgreen']
    titles = ['Blur', 'Expression', 'Illumination', 'Occlusion', 'Pose', 'Invalid']
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    axs = axs.flatten()

    for i, (key, value) in enumerate(data.items()):
        categories = sorted(value.keys())
        counts = [value[cat] for cat in categories]
        labels = [str(cat) for cat in categories]
        axs[i].bar(labels, counts, color=colors[:len(counts)])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Categories')
        
        text = "Counts" if i != 0 else ""
        axs[i].set_ylabel(text)
        
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()


def draw_predicted_bboxes(
    dataset: DatasetDict, original: pd.DataFrame, preds: pd.DataFrame, 
    n_images: int = 4,
):
    
    img_ids = random.sample(original['img_id'].unique().tolist(), n_images)
    fig, axs = plt.subplots(n_images, 2, figsize=(6, n_images * 2))
    coords = ["xmin", "ymin", "width", "height"]
    
    for irow, img_id in enumerate(img_ids):
        img = dataset['validation'][img_id]['image']
        bboxes = {
            'Original bbox': (
                'green',
                original.query(f"img_id == {img_id}")[coords].values,
            ),
            'Predicted bbox': (
                'red',
                preds.query(f"img_id == {img_id}")[coords].values,
            )
        }
        
        for icol, (title, (color, faces)) in enumerate(bboxes.items()):    
            axs[irow, icol].imshow(img)
            axs[irow, icol].set_title(title)
            axs[irow, icol].axis('off')
            
            for (x, y, width, height) in faces:
                rect = patches.Rectangle(
                    (x, y), width, height, 
                    linewidth=2, edgecolor=color, facecolor='none',
                )
                axs[irow, icol].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('../data/preds.png')
    plt.show()