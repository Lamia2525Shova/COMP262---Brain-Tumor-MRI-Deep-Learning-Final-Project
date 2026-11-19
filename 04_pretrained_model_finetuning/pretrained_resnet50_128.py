# -*- coding: utf-8 -*-
"""
Created on Sat Nov 7 2026

@author: Haripriya Gounalan
This script trains and evaluates ResNet50 three ways:
  Stage 1 - transfer learning, feature extraction (ImageNet weights frozen, only the new head is trained)
  Stage 2 - transfer learning, fine-tuning (last ResNet50 block conv5_x unfrozen, very small learning rate)
  Scratch - same architecture with random weights, every layer trained
Runs at 128x128 (the input size of our CNNs); pretrained_resnet50_224.py runs the same code at 224x224.
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

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from data_exploration import SEED, classes, section, show, testing_folder_path, training_folder_path
from model_utils import ModelTrainer, data_augmentation, free_gpu_memory
from preprocessing import preprocess_dataset

# ============================================================
# Configuration
# ============================================================
batch_size = 32
patience   = 5

# Stage 1: feature extraction - the new head starts from random weights, so a normal learning rate
stage1_epochs        = 10
stage1_learning_rate = 1e-3

# Stage 2: fine-tuning - very small steps, so the pretrained weights are only adjusted
stage2_epochs        = 20
stage2_learning_rate = 1e-5
fine_tune_from       = "conv5_"           # unfreeze ResNet50 layers whose name starts with this (last block)

# Scratch: random weights, every layer trained
scratch_epochs        = 30
scratch_learning_rate = 1e-3

keras.utils.set_random_seed(SEED)

# ============================================================
# Build ResNet50
# ============================================================
def build_resnet50(weights, name, image_size):
    """ResNet50 base + GlobalAveragePooling2D + Dropout + softmax output.

    weights="imagenet": transfer learning; the base runs with training=False, so its BatchNormalization
                        layers keep the ImageNet statistics (also when unfrozen in Stage 2).
    weights=None      : from scratch; BatchNormalization learns its statistics from the MRI images.
    """
    base = ResNet50(weights=weights, include_top=False, input_shape=(*image_size, 3))

    inputs  = Input(shape=(*image_size, 3))
    x       = data_augmentation()(inputs)                           # rotation, shift, flip (training only)
    x       = base(x, training=False) if weights else base(x)
    x       = GlobalAveragePooling2D()(x)                           # one value per feature map
    x       = Dropout(0.4)(x)                                       # reduce overfitting
    outputs = Dense(len(classes), activation="softmax")(x)          # probabilities for the 4 classes

    return Model(inputs=inputs, outputs=outputs, name=name)


# ============================================================
# Freeze / Unfreeze ResNet50 Layers
# ============================================================
def freeze_base(model):
    """Stage 1: freeze every ResNet50 layer (use it only as a feature extractor)."""
    model.get_layer("resnet50").trainable = False


def unfreeze_last_block(model):
    """Stage 2: unfreeze only the last ResNet50 block (conv5_x); earlier blocks stay frozen."""
    base = model.get_layer("resnet50")
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith(fine_tune_from)


def count_trainable(model):
    """Number of trainable parameters (shows what each stage actually trains)."""
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ============================================================
# Train and Evaluate One Model
# ============================================================
def train_and_evaluate(model, model_name, learning_rate, epochs, data, display_images, results_dir, batch_size=batch_size):
    """Train with early stopping + checkpoint, evaluate on test data, return the trainer and test score."""
    x_train, y_train, x_val, y_val, x_test, y_test = data

    trainer = ModelTrainer(model, model_name,
                           x_train, y_train,
                           x_val,   y_val,
                           x_test,  y_test,
                           class_names=classes, class_weight="balanced",
                           optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           batch_size=batch_size, no_epochs=epochs, patience=patience,
                           display_images=display_images,           # original images, not the ResNet-scaled ones
                           results_dir=results_dir,
                           verbosity=2)
    trainer.train()
    score, _ = trainer.evaluate()
    return trainer, score


# ============================================================
# Plot Model Comparison
# ============================================================
def plot_resnet_comparison(model_names, histories, scores, save_path, title):
    """Left: test accuracy per model. Right: validation accuracy per epoch (Stage 2 continues after Stage 1)."""
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]                      # one fixed color per model
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor("white")

    # test accuracy bars
    accuracies = [s["accuracy"] for s in scores]
    bars = ax_bar.barh(model_names, accuracies, color=colors, height=0.55)
    ax_bar.bar_label(bars, labels=[f"{a:.1%}" for a in accuracies], padding=6, color=ink, fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.12)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_bar.set_title("Test accuracy", loc="left", fontsize=13, color=ink, pad=12)

    # validation accuracy curves
    start = 0
    for name, history, color in zip(model_names, histories, colors):
        val_acc = history.history["val_accuracy"]
        offset  = start if name.startswith("Stage 2") else 0        # Stage 2 picks up where Stage 1 stopped
        epochs  = range(offset + 1, offset + len(val_acc) + 1)
        ax_line.plot(epochs, val_acc, color=color, linewidth=2, label=name)
        ax_line.plot(epochs[-1], val_acc[-1], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
        if name.startswith("Stage 1"):
            start = len(val_acc)

    ax_line.set_ylim(0, 1.02)
    ax_line.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_line.set_xlabel("Epoch", color=muted)
    ax_line.set_title("Validation accuracy per epoch", loc="left", fontsize=13, color=ink, pad=12)
    ax_line.legend(frameon=False, loc="lower right")

    # shared clean style: no box, light grid behind the data
    for ax in (ax_bar, ax_line):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.grid(axis="x" if ax is ax_bar else "y", color=grid)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)

    fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=ink)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Run the Full Experiment at One Image Size
# ============================================================
def run_experiment(image_size=(128, 128), batch_size=batch_size):
    """Steps 1-6 at the given image size; results go to results/pretrained_resnet50_<size>/.
    batch_size: smaller for large images, so training fits in GPU memory."""
    results_dir = os.path.join(_here, "results", f"pretrained_resnet50_{image_size[0]}")
    os.makedirs(results_dir, exist_ok=True)

    # ============================================================
    # Step 1: Load and Preprocess Data
    # ============================================================
    section("Step 1: Load and Preprocess Data")

    train_images, train_labels = preprocess_dataset(training_folder_path, size=image_size)     # crop, resize to image_size
    test_images,  test_labels  = preprocess_dataset(testing_folder_path,  size=image_size)

    # ResNet50 expects its own ImageNet scaling (RGB -> BGR, minus the ImageNet mean), not / 255
    x_train_full = preprocess_input(train_images.astype("float32"))
    x_test       = preprocess_input(test_images.astype("float32"))
    y_train_full = to_categorical(train_labels, num_classes=len(classes))
    y_test       = to_categorical(test_labels,  num_classes=len(classes))

    # ============================================================
    # Step 2: Split Train and Validation Data
    # ============================================================
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.20, stratify=train_labels, random_state=SEED)
    del x_train_full, train_images                          # free RAM: large float images

    section("Step 2: Split Train and Validation Data")

    show("Dataset splits", pd.DataFrame({
        "Images": [len(x_train), len(x_val), len(x_test)],
        "Image shape": [x_train.shape[1:], x_val.shape[1:], x_test.shape[1:]],
        "Labels shape": [y_train.shape[1:], y_val.shape[1:], y_test.shape[1:]]},
        index=["Training", "Validation", "Testing"]))

    data = (x_train, y_train, x_val, y_val, x_test, y_test)

    # ============================================================
    # Step 3: Stage 1 - Feature Extraction (ResNet50 frozen)
    # ============================================================
    section("Step 3: Stage 1 - Feature Extraction (ResNet50 frozen)")

    stage1_model = build_resnet50("imagenet", "ResNet50_Stage1_Frozen", image_size)
    freeze_base(stage1_model)
    stage1_params = count_trainable(stage1_model)                  # recorded now: Stage 2 unfreezes this model
    print(f"Trainable parameters: {stage1_params:,} (classification head only)")

    stage1_trainer, stage1_score = train_and_evaluate(
        stage1_model, "ResNet50 Stage1 Frozen", stage1_learning_rate, stage1_epochs, data, test_images, results_dir, batch_size)

    # ============================================================
    # Step 4: Stage 2 - Fine-Tuning (conv5_x unfrozen)
    # ============================================================
    section("Step 4: Stage 2 - Fine-Tuning (conv5_x unfrozen)")

    stage2_model = stage1_trainer.model                             # continue from the best Stage 1 model
    unfreeze_last_block(stage2_model)
    stage2_params = count_trainable(stage2_model)
    print(f"Trainable parameters: {stage2_params:,} (head + conv5_x block)")

    stage2_trainer, stage2_score = train_and_evaluate(
        stage2_model, "ResNet50 Stage2 Fine Tuned", stage2_learning_rate, stage2_epochs, data, test_images, results_dir, batch_size)

    # Stage 1 and 2 are done (scores and histories are kept): remove their models from GPU memory
    stage1_trainer.model = stage2_trainer.model = None
    del stage1_model, stage2_model
    free_gpu_memory()

    # ============================================================
    # Step 5: ResNet50 from Scratch (random weights)
    # ============================================================
    section("Step 5: ResNet50 from Scratch (random weights)")

    scratch_model  = build_resnet50(None, "ResNet50_From_Scratch", image_size)
    scratch_params = count_trainable(scratch_model)
    print(f"Trainable parameters: {scratch_params:,} (all layers)")

    scratch_trainer, scratch_score = train_and_evaluate(
        scratch_model, "ResNet50 From Scratch", scratch_learning_rate, scratch_epochs, data, test_images, results_dir, batch_size)

    # ============================================================
    # Step 6: Compare the Three ResNet50 Models
    # ============================================================
    section("Step 6: Compare the Three ResNet50 Models")

    trainers = [stage1_trainer, stage2_trainer, scratch_trainer]
    scores   = [stage1_score,   stage2_score,   scratch_score]

    comparison_df = pd.DataFrame({
        "Model": ["Stage 1: Feature Extraction", "Stage 2: Fine-Tuned", "From Scratch"],
        "Trainable Params": [stage1_params, stage2_params, scratch_params],
        "Epochs Run": [len(t.history.history["loss"]) for t in trainers],
        "Test Accuracy": [s["accuracy"] for s in scores],
        "Test Loss": [s["loss"] for s in scores]})
    comparison_df.to_csv(os.path.join(results_dir, "resnet50_comparison.csv"), index=False)

    display_df = comparison_df.set_index("Model").copy()
    display_df["Trainable Params"] = display_df["Trainable Params"].map(lambda value: f"{value:,}")
    display_df["Test Accuracy"]    = display_df["Test Accuracy"].map(lambda value: f"{value:.2%}")
    display_df["Test Loss"]        = display_df["Test Loss"].map(lambda value: f"{value:.4f}")
    show(f"ResNet50 ({image_size[0]}x{image_size[1]}): feature extraction vs fine-tuning vs from scratch", display_df)

    plot_resnet_comparison(list(comparison_df["Model"]), [t.history for t in trainers], scores,
                           os.path.join(results_dir, "resnet50_comparison.png"),
                           f"ResNet50 at {image_size[0]}x{image_size[1]}: Feature Extraction vs Fine-Tuning vs From Scratch")


if __name__ == "__main__":
    run_experiment((128, 128))
