# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:40:02 2026

@author: Haripriya Gounalan
"""

import os
import sys

# Project setup: find the project root, make the shared folders importable and run from the root,
# so "./data" works from any folder (also on Kaggle, where all files are in one folder); results are saved next to each script
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _here if os.path.exists(os.path.join(_here, "model_utils.py")) else os.path.dirname(_here)
sys.path[:0] = [os.path.join(PROJECT_ROOT, d) for d in ("01_data_exploration_preprocessing", "utils")]
os.chdir(PROJECT_ROOT)

import hashlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.utils import shuffle

# ============================================================
# Configuration
# ============================================================
SEED = 42

training_folder_path = "./data/Training/"
testing_folder_path  = "./data/Testing/"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Hardware Configuration
# ============================================================
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Helper Functions
# ============================================================
def section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def show(title, df):
    """Print a titled, indented table."""
    print(f"\n  {title}\n  {'-' * 66}")
    print("\n".join("    " + line for line in df.rename_axis(index=None, columns=None).to_string().splitlines()))


def load_images(dataset_path):
    """Load original MRI images at their original size (a list, since sizes differ; uses several GB of RAM)."""
    images, labels = [], []

    for label, folder in enumerate(classes):
        for filename in sorted(os.listdir(os.path.join(dataset_path, folder))):
            img = cv2.imread(os.path.join(dataset_path, folder, filename))          # Read image (BGR)
            if img is None:
                continue

            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                     # Temporarily store as RGB without resizing
            labels.append(label)

    return images, np.asarray(labels, dtype=np.int64)


def is_gray(img):
    """True if every pixel has R = G = B (no real color)."""
    return bool(np.all((img[..., 0] == img[..., 1]) & (img[..., 1] == img[..., 2])))


def inspect_raw_images(dataset_path):
    """Read only the header of each original file and return its size and color mode."""
    records = []
    for folder in classes:
        for filename in sorted(os.listdir(os.path.join(dataset_path, folder))):
            with Image.open(os.path.join(dataset_path, folder, filename)) as im:        # Header only, no pixel decode
                records.append({"class": folder, "size": f"{im.width} x {im.height}",
                                "square": im.width == im.height, "mode": im.mode})
    return pd.DataFrame(records)


def file_fingerprints(dataset_paths):
    """MD5 fingerprint of every file: identical fingerprints = byte-for-byte identical (duplicate) images."""
    records = []
    for split, dataset_path in dataset_paths.items():
        for folder in classes:
            for filename in sorted(os.listdir(os.path.join(dataset_path, folder))):
                with open(os.path.join(dataset_path, folder, filename), "rb") as file:
                    records.append({"split": split, "class": folder, "file": filename,
                                    "md5": hashlib.md5(file.read()).hexdigest()})
    return pd.DataFrame(records)


def plot_class_distribution(dataset_name, labels, class_names):
    """Plot the class distribution of a Brain Tumor MRI dataset."""
    counts  = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar_label(ax.bar(class_names, counts, color="#F4A261", width=0.6), padding=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5")
    ax.set_axisbelow(True)
    ax.set(ylabel="Number of images", xlabel="Classes", title=f"{dataset_name} class distribution")

    plt.tight_layout()
    plt.show()


def plot_sample_images(dataset_name, images, labels, class_names, num_images=12):
    """Display MRI samples in an adaptive grid."""
    n    = min(num_images, len(images))
    cols = 5 if n >= 15 else 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))              # Adaptive grid layout
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]], fontsize=10)

    fig.suptitle(f"{dataset_name} MRI Sample Images", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"PyTorch compute device: {compute_device}")

    # ============================================================
    # Step 1.a: Load the Dataset (original sizes, no preprocessing)
    # ============================================================
    section("Step 1.a: Load the dataset")

    try:
        x_train, y_train = load_images(training_folder_path)
        x_test,  y_test  = load_images(testing_folder_path)
        print(f" Brain Tumor MRI Dataset loaded successfully.")

    except Exception as e:
        print(f"Error loading Brain Tumor MRI dataset: {e}")
        raise

    # Shuffle the datasets
    x_train, y_train = shuffle(x_train, y_train, random_state=SEED)
    x_test,  y_test  = shuffle(x_test,  y_test,  random_state=SEED)

    # ============================================================
    # Step 1.b: Inspect the Original Files (sizes and color modes)
    # ============================================================
    section("Step 1.b: Inspect the original files (sizes and color modes)")

    raw = {"Training": inspect_raw_images(training_folder_path), "Testing": inspect_raw_images(testing_folder_path)}

    show("Original image sizes", pd.DataFrame({
        split: {"Distinct sizes": df["size"].nunique(), "Square": df["square"].sum(), "Non-square": (~df["square"]).sum()}
        for split, df in raw.items()}))

    for split, df in raw.items():
        show(f"Most common sizes - {split}", df["size"].value_counts().head(5).to_frame("images"))

    for split, df in raw.items():
        show(f"Color modes per class - {split}  (L = grayscale, P = palette)",
             pd.crosstab(df["class"], df["mode"], margins=True, margins_name="Total"))

    # ============================================================
    # Step 1.c: Data Exploration
    # ============================================================
    section("Step 1.c: Data Exploration")

    gray = {"Training": np.array([is_gray(i) for i in x_train]), "Testing": np.array([is_gray(i) for i in x_test])}
    y    = {"Training": y_train, "Testing": y_test}

    show("Loaded images", pd.DataFrame({split: {"Images": len(x), "Distinct shapes": len({i.shape for i in x}),
                "Pixel range": f"{min(i.min() for i in x)} - {max(i.max() for i in x)}", "R = G = B": f"{gray[split].mean():.1%}"}
        for split, x in [("Training", x_train), ("Testing", x_test)]}))

    show("Class distribution", pd.DataFrame({"Training": np.bincount(y_train), "Testing": np.bincount(y_test)}, index=classes))

    show("Images with R = G = B (no real color) per class", pd.DataFrame(
        {split: [f"{gray[split][y[split] == c].mean():.1%}" for c in range(len(classes))] for split in gray}, index=classes))

    # ============================================================
    # Step 1.d: Duplicate Images (byte-for-byte identical files)
    # ============================================================
    section("Step 1.d: Duplicate images")

    files = file_fingerprints({"Training": training_folder_path, "Testing": testing_folder_path})
    dups  = files[files.duplicated("md5", keep=False)]                          # every copy of a repeated image

    show("Duplicates per split", pd.DataFrame({split: {
        "Files": (files["split"] == split).sum(),
        "Duplicate groups": dups.loc[dups["split"] == split, "md5"].nunique(),
        "Extra copies": dups[dups["split"] == split].duplicated("md5").sum()}  # copies beyond the first
        for split in ["Training", "Testing"]}))

    per_group = dups.groupby("md5").agg(splits=("split", "nunique"), labels=("class", "nunique"))
    show("Duplicate checks", pd.DataFrame({"Groups": {
        "Shared between Training and Testing (leakage)": (per_group["splits"] > 1).sum(),
        "Copies with different labels (label conflict)": (per_group["labels"] > 1).sum()}}))

    show("Extra copies per class", pd.DataFrame({split: dups[dups["split"] == split]
        .loc[lambda d: d.duplicated("md5"), "class"].value_counts().reindex(classes, fill_value=0)
        for split in ["Training", "Testing"]}))

    # Plot class distribution
    plot_class_distribution("Training", y_train, classes)
    plot_class_distribution("Testing",  y_test,  classes)

    # Plot sample images
    plot_sample_images("Training", x_train, y_train, classes, num_images=12)
