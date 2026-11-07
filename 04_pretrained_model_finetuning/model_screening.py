# -*- coding: utf-8 -*-
"""
Created on Sat Nov 7 2026

@author: Haripriya Gounalan
This script screens four pretrained state-of-the-art models to choose one for the full experiment.
Every candidate is fine-tuned the same way and compared on VALIDATION data only (the test set is not loaded):
  Stage 1 - base frozen, only the new classification head is trained
  Stage 2 - top 20% of the base layers unfrozen, small learning rate
"""

import os
import sys
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation

# Project setup: find the project root, make the shared folders importable and run from the root,
# so "./data" works from any folder (also on Kaggle, where all files are in one folder); results are saved next to each script
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _here if os.path.exists(os.path.join(_here, "model_utils.py")) else os.path.dirname(_here)
sys.path[:0] = [os.path.join(PROJECT_ROOT, d) for d in ("01_data_exploration_preprocessing", "utils")]
os.chdir(PROJECT_ROOT)

import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import applications
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from data_exploration import SEED, classes, section, show, training_folder_path
from model_utils import ModelTrainer, data_augmentation, free_gpu_memory
from preprocessing import preprocess_dataset

# ============================================================
# Configuration
# ============================================================
image_size  = (128, 128)                  # same input size for every candidate and for our CNNs (fair comparison)
batch_size  = 32
patience    = 5
results_dir = os.path.join(_here, "results", "model_screening")

stage1_epochs, stage1_learning_rate = 5,  1e-3      # head only
stage2_epochs, stage2_learning_rate = 10, 1e-5      # top layers unfrozen
fine_tune_fraction = 0.2                            # unfreeze the top 20% of each base model's layers

# candidate name -> (model constructor, its own ImageNet preprocessing)
CANDIDATES = {
    "ResNet50":       (applications.ResNet50,       applications.resnet50.preprocess_input),      # skip connections
    "Xception":       (applications.Xception,       applications.xception.preprocess_input),      # depthwise separable convs
    "EfficientNetB0": (applications.EfficientNetB0, applications.efficientnet.preprocess_input),  # compound scaling
    "DenseNet121":    (applications.DenseNet121,    applications.densenet.preprocess_input),      # dense connections
}

keras.utils.set_random_seed(SEED)

# ============================================================
# Build a Pretrained Model + Classification Head
# ============================================================
def build_pretrained(constructor, name):
    """Pretrained base (frozen, BatchNorm in inference mode) + GlobalAveragePooling2D + Dropout + softmax."""
    base = constructor(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    base.trainable = False

    inputs  = Input(shape=(*image_size, 3))
    x       = data_augmentation()(inputs)
    x       = base(x, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dropout(0.4)(x)
    outputs = Dense(len(classes), activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name=name), base.name


def unfreeze_top(base, fraction):
    """Make only the top `fraction` of the base model's layers trainable (same rule for every candidate)."""
    base.trainable = True
    cut = int(len(base.layers) * (1 - fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut


def count_trainable(model):
    """Number of trainable parameters."""
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ============================================================
# Fine-Tune One Candidate (Stage 1 + Stage 2)
# ============================================================
def screen_candidate(name, constructor, preprocess, train_u8, y_train, val_u8, y_val):
    """Fine-tune one candidate and return its validation results (test data is never used here)."""
    section(f"Screening {name}")
    x_train = preprocess(train_u8.astype("float32"))        # each model expects its own ImageNet scaling
    x_val   = preprocess(val_u8.astype("float32"))
    start   = time.perf_counter()

    def train_stage(model, stage, epochs, learning_rate):
        trainer = ModelTrainer(model, f"{name} {stage}",
                               x_train, y_train,
                               x_val,   y_val,
                               x_val,   y_val,                       # placeholder: evaluate() is not called
                               optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                               batch_size=batch_size, no_epochs=epochs, patience=patience, class_weight="balanced",
                               results_dir=os.path.join(results_dir, name), verbosity=2)
        trainer.train()
        score = trainer.model.evaluate(x_val, y_val, verbose=0, return_dict=True)
        return trainer, score

    # Stage 1: frozen base
    model, base_name = build_pretrained(constructor, name)
    stage1_trainer, stage1_score = train_stage(model, "Stage1", stage1_epochs, stage1_learning_rate)

    # Stage 2: continue from the best Stage 1 model with the top layers unfrozen
    model = stage1_trainer.model
    unfreeze_top(model.get_layer(base_name), fine_tune_fraction)
    trainable = count_trainable(model)
    stage2_trainer, stage2_score = train_stage(model, "Stage2", stage2_epochs, stage2_learning_rate)

    row = {"Model": name,
           "Total Params": model.count_params(),
           "Stage 2 Trainable": trainable,
           "Stage 1 Val Accuracy": stage1_score["accuracy"],
           "Stage 2 Val Accuracy": stage2_score["accuracy"],
           "Stage 2 Val Loss": stage2_score["loss"],
           "Epochs Run": len(stage1_trainer.history.epoch) + len(stage2_trainer.history.epoch),
           "Minutes": (time.perf_counter() - start) / 60}

    del model, stage1_trainer, stage2_trainer                # x_train / x_val are freed when the function returns
    free_gpu_memory()
    return row


# ============================================================
# Plot the Screening Results
# ============================================================
def plot_screening(results_df, save_path):
    """Validation accuracy after Stage 1 and Stage 2 for every candidate (best on top)."""
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    names = list(results_df["Model"])
    y     = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 1.2 * len(names) + 1.5))
    for offset, column, color, label in [(-0.18, "Stage 1 Val Accuracy", "#86b6ef", "Stage 1: frozen"),
                                         (0.18,  "Stage 2 Val Accuracy", "#2a78d6", "Stage 2: fine-tuned")]:
        values = results_df[column]
        bars   = ax.barh(y + offset, values, height=0.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=5, color=ink, fontsize=10)

    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.grid(axis="x", color=grid)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Model screening: validation accuracy at {image_size[0]}x{image_size[1]}",
                 loc="left", fontsize=14, color=ink, pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)

    # ============================================================
    # Step 1: Load, Preprocess and Split the Training Data (test data is not loaded)
    # ============================================================
    section("Step 1: Load and Split the Training Data")

    train_images, train_labels = preprocess_dataset(training_folder_path, size=image_size)
    train_u8, val_u8, y_train, y_val = train_test_split(
        train_images, to_categorical(train_labels, num_classes=len(classes)),
        test_size=0.20, stratify=train_labels, random_state=SEED)                  # same split as the other scripts
    del train_images

    show("Dataset splits", pd.DataFrame({"Images": [len(train_u8), len(val_u8)],
                                         "Image shape": [train_u8.shape[1:], val_u8.shape[1:]]},
                                        index=["Training", "Validation"]))

    # ============================================================
    # Step 2: Fine-Tune Every Candidate the Same Way
    # ============================================================
    rows = [screen_candidate(name, constructor, preprocess, train_u8, y_train, val_u8, y_val)
            for name, (constructor, preprocess) in CANDIDATES.items()]

    # ============================================================
    # Step 3: Compare and Select the Best Model (validation accuracy)
    # ============================================================
    section("Step 3: Compare and Select the Best Model")

    results_df = pd.DataFrame(rows).sort_values("Stage 2 Val Accuracy", ascending=False)
    results_df.to_csv(os.path.join(results_dir, "model_screening.csv"), index=False)

    display_df = results_df.set_index("Model").copy()
    for column in ["Total Params", "Stage 2 Trainable"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:,}")
    for column in ["Stage 1 Val Accuracy", "Stage 2 Val Accuracy"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.2%}")
    display_df["Stage 2 Val Loss"] = display_df["Stage 2 Val Loss"].map(lambda value: f"{value:.4f}")
    display_df["Minutes"]          = display_df["Minutes"].map(lambda value: f"{value:.1f}")
    show("Model screening (validation data)", display_df)

    plot_screening(results_df, os.path.join(results_dir, "model_screening.png"))
    print(f"\nSelected model: {results_df.iloc[0]['Model']} (highest Stage 2 validation accuracy)")
