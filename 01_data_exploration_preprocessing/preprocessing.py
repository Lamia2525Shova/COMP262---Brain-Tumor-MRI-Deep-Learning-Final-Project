# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:09:36 2026

@author: Haripriya Gounalan

This script implements the preprocessing steps for the MRI dataset, including color mode normalization, cropping black margins, 
resizing images, normalizing pixel values, and one-hot encoding labels. It also provides visualization of the preprocessing results.
"""
import os
import sys
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import hashlib
import os
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical

from data_exploration import classes, plot_class_distribution, section, show, testing_folder_path, training_folder_path

# ============================================================
# Configuration
# ============================================================
image_size = (128, 128)


# ============================================================
# Helper Functions
# ============================================================
def crop_box(img):
    """Return (x, y, w, h) of the largest contour (the brain), or the full image if none is found."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3, 3), 0)           # Blur to reduce noise
    _, mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)                           # Binary mask of brain region
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)          # Remove noise and fill gaps
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Find outer contours

    if not contours:
        return 0, 0, img.shape[1], img.shape[0]
    return cv2.boundingRect(max(contours, key=cv2.contourArea))                         # Bounding box of largest contour


def crop_img(img):
    """Crop the image to the brain region, removing the black margins."""
    x, y, w, h = crop_box(img)
    return img[y:y + h, x:x + w] if w and h else img


def preprocess_dataset(dataset_path, crop=True, size=image_size, remove_duplicates=True):
    """Read every image and apply steps 1-4; returns (images, labels) as uint8 / int arrays.
    size: output size (default 128x128; ResNet50 uses 224x224, its original ImageNet size).
    remove_duplicates: keep only the first copy of byte-for-byte identical files (Training and Testing)."""
    images, labels, seen, removed = [], [], set(), 0

    for label, folder in enumerate(classes):
        for filename in sorted(os.listdir(os.path.join(dataset_path, folder))):
            path = os.path.join(dataset_path, folder, filename)
            if remove_duplicates:
                with open(path, "rb") as file:
                    md5 = hashlib.md5(file.read()).hexdigest()                          # same fingerprint = same image
                if md5 in seen:
                    removed += 1
                    continue
                seen.add(md5)

            img = cv2.imread(path)                                                      # 1. Any color mode -> 3-channel BGR
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                  # 2. BGR -> RGB
            img = crop_img(img) if crop else img                                        # 3. Remove black margins
            images.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))          # 4. Resize to size
            labels.append(label)

    if remove_duplicates:
        print(f"    Duplicates removed from {dataset_path}: {removed}")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.int64)


def normalize_and_encode(images, labels):
    """Steps 5-6: pixels -> float32 in [0, 1], labels -> one-hot vectors (needed by categorical_crossentropy)."""
    return images.astype("float32") / 255, to_categorical(labels, num_classes=len(classes))


def plot_before_after(dataset_path, processed):
    """Original (with the crop box drawn) vs preprocessed image, for the first image of each class."""
    fig, axes = plt.subplots(2, len(classes), figsize=(13, 7), facecolor="white")

    for i, folder in enumerate(classes):
        filename = sorted(os.listdir(os.path.join(dataset_path, folder)))[0]           # Same file as processed[i]
        original = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, folder, filename)), cv2.COLOR_BGR2RGB)
        x, y, w, h = crop_box(original)

        top, bottom = axes[0, i], axes[1, i]
        top.imshow(original)
        top.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="#2A9D8F", linewidth=2, linestyle="--"))
        top.set_title(folder.capitalize(), fontsize=13, fontweight="bold", pad=10)
        top.set_xlabel(f"{original.shape[1]} x {original.shape[0]} px", fontsize=9, color="#555555")

        bottom.imshow(processed[i])
        bottom.set_xlabel(f"{image_size[0]} x {image_size[1]} px", fontsize=9, color="#555555")

        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#DDDDDD")

    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Preprocessed", fontsize=12, fontweight="bold", labelpad=10)

    fig.suptitle("Before vs After Preprocessing", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.925, "Dashed box = region kept by crop_img()  ·  then resized to "
             f"{image_size[0]} x {image_size[1]}", ha="center", fontsize=10, color="#555555")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


# ============================================================
# Step 1: Preprocessing
# ============================================================
if __name__ == "__main__":
    section("Step 1: Preprocessing")
    print("    1. Mixed color modes : cv2.imread()            -> every file becomes 3-channel BGR")
    print("    2. BGR channel order : cv2.cvtColor(BGR2RGB)   -> channels reordered to RGB")
    print("    3. Black margins     : crop_img()              -> cropped to the brain region")
    print(f"    4. Different sizes   : cv2.resize(INTER_AREA)  -> every image becomes {image_size[0]} x {image_size[1]}")
    print("    5. Pixel scale       : astype(float32) / 255   -> pixels in [0, 1]")
    print("    6. Labels            : to_categorical()        -> one-hot vectors")

    x_train, y_train = preprocess_dataset(training_folder_path)
    x_test,  y_test  = preprocess_dataset(testing_folder_path)

    show("Class distribution after removing duplicates (Training and Testing)", pd.DataFrame(
        {"Training": np.bincount(y_train), "Testing": np.bincount(y_test)}, index=classes)
        .pipe(lambda df: pd.concat([df, df.sum().to_frame("Total").T])))

    # plot class distribution after removing duplicates (Training and Testing)
    plot_class_distribution("Training", y_train, classes)
    plot_class_distribution("Testing",  y_test,  classes)

    # plot before vs after (uses the uint8 images, before normalization)
    first = [np.flatnonzero(y_train == i)[0] for i in range(len(classes))]             # First image of each class
    plot_before_after(training_folder_path, x_train[first])

    labels_before    = y_train[first]
    x_train, y_train = normalize_and_encode(x_train, y_train)
    x_test,  y_test  = normalize_and_encode(x_test,  y_test)

    show("Label encoding (one example per class)", pd.DataFrame(
        {"Label": labels_before, "One-hot": [str(v.astype(int)) for v in y_train[first]]}, index=classes))

    show("Result after preprocessing", pd.DataFrame({
        split: {"Images": x.shape[0], "Image shape": x.shape[1:], "Dtype": x.dtype,
                "Pixel range": f"{x.min():.2f} - {x.max():.2f}", "Labels shape": y.shape}
        for split, x, y in [("Training", x_train, y_train), ("Testing", x_test, y_test)]}))
