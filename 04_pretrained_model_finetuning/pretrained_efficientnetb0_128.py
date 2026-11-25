# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:12:45 2026

@author: Abdallah Divker
"""
"""
EfficientNetB0 experiments for Brain Tumor MRI Classification.

Experiments:
    1. ImageNet pretrained - frozen backbone (feature extraction)
    2. ImageNet pretrained - fine-tune final EfficientNet stage
    3. Random initialization - train from scratch

Uses the existing project preprocessing pipeline.
"""

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# Project Setup
# ============================================================

_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_here)

sys.path[:0] = [
    os.path.join(PROJECT_ROOT, "01_data_exploration_preprocessing"),
    os.path.join(PROJECT_ROOT, "utils")
]

os.chdir(PROJECT_ROOT)

import time
import gc

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from keras.applications import EfficientNetB0
from keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)
from keras.models import Model
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_exploration import (
    SEED,
    classes,
    section,
    show,
    training_folder_path,
    testing_folder_path
)

from preprocessing import preprocess_dataset
from model_utils import (
    data_augmentation,
    balanced_class_weights,
    free_gpu_memory
)


# ============================================================
# Configuration
# ============================================================

image_size = (128, 128)

batch_size = 32
patience = 5

# Feature extraction
stage1_epochs = 10
stage1_learning_rate = 1e-3

# Fine-tuning
stage2_epochs = 20
stage2_learning_rate = 1e-5

# Training from scratch
scratch_epochs = 30
scratch_learning_rate = 1e-3

keras.utils.set_random_seed(SEED)

RESULTS_DIR = os.path.join(
    _here,
    "results",
    "pretrained_efficientnetb0_128"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Hardware Verification
# ============================================================

def check_gpu():
    print("\nHardware configuration")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Check the PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Keras backend:", keras.backend.backend())


# ============================================================
# Build EfficientNetB0
# ============================================================

def build_efficientnetb0(weights, name, image_size):
    """
    EfficientNetB0 backbone + classification head.

    EfficientNetB0 includes internal input rescaling.
    Therefore, inputs must remain in the 0-255 range.

    For pretrained models, training=False preserves
    pretrained BatchNormalization statistics.
    """

    base = EfficientNetB0(
        weights=weights,
        include_top=False,
        input_shape=(*image_size, 3)
    )

    inputs = Input(shape=(*image_size, 3))

    x = data_augmentation()(inputs)

    if weights is not None:
        x = base(x, training=False)
    else:
        x = base(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.4)(x)

    outputs = Dense(
        len(classes),
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=name
    )

    return model


# ============================================================
# Freeze / Unfreeze Backbone
# ============================================================

def freeze_base(model):
    """Freeze the complete EfficientNetB0 backbone."""

    base = model.get_layer("efficientnetb0")
    base.trainable = False


def unfreeze_last_block(model):
    """
    Unfreeze the final EfficientNetB0 MBConv stage
    and final convolutional feature layers.

    BatchNormalization layers remain frozen.
    """

    base = model.get_layer("efficientnetb0")
    base.trainable = True

    trainable_prefixes = (
        "block7a_",
        "top_conv",
        "top_activation"
    )

    unfrozen_layers = []

    for layer in base.layers:

        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

        elif layer.name.startswith(trainable_prefixes):
            layer.trainable = True
            unfrozen_layers.append(layer.name)

        else:
            layer.trainable = False

    if not unfrozen_layers:
        raise RuntimeError(
            "Could not identify the final EfficientNetB0 layers. "
            "Inspect base.layers before proceeding."
        )

    print("\nUnfrozen EfficientNetB0 layers:")

    for name in unfrozen_layers:
        print(" ", name)


def count_trainable(model):
    """Count trainable parameters."""

    return sum(
        int(np.prod(weight.shape))
        for weight in model.trainable_weights
    )


# ============================================================
# Train Using Validation Data
# ============================================================

def train_model(
    model,
    model_name,
    learning_rate,
    epochs,
    data,
    results_dir
):
    """
    Train using validation data only.

    Test data is deliberately excluded from this function.
    """

    x_train, y_train, x_val, y_val = data

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(
        results_dir,
        f"{model_name.lower()}_best.keras"
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n" + "=" * 65)
    print(f"Training: {model_name}")
    print("=" * 65)

    model.summary()

    start_time = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=balanced_class_weights(y_train),
        callbacks=callbacks,
        verbose=2
    )

    # Save epoch-by-epoch training history
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    history_path = os.path.join(
        results_dir,
        f"{model_name.lower()}_training_history.csv"
    )

    history_df.to_csv(history_path, index=False)

    print(f"Training history saved: {history_path}")

    training_time = (time.perf_counter() - start_time) / 60

    # Load the checkpoint selected using validation loss.
    best_model = keras.models.load_model(checkpoint_path)

    best_epoch = int(
        np.argmin(history.history["val_loss"])
    ) + 1

    print(f"\nBest epoch: {best_epoch}")
    print(f"Training time: {training_time:.2f} minutes")

    return (
        best_model,
        history,
        training_time,
        best_epoch
    )


# ============================================================
# Final Test Evaluation
# ============================================================

def evaluate_test(
    model,
    model_name,
    x_test,
    y_test,
    results_dir
):
    """
    Evaluate the finalized model on the held-out test set.
    """

    predictions = model.predict(
        x_test,
        batch_size=batch_size,
        verbose=0
    )

    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    metrics = {
        "Model": model_name,

        "Accuracy": accuracy_score(
            true_labels,
            predicted_labels
        ),

        "Precision": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        ),

        "Recall": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        ),

        "Macro F1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )
    }

    print("\nTest results:")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    # Classification report
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    with open(
        os.path.join(
            results_dir,
            f"{model_name.lower()}_classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # Confusion matrix
    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("\nConfusion Matrix:")
    print(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))

    import seaborn as sns

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )

    ax.set(
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{model_name} - Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            f"{model_name.lower()}_confusion_matrix.png"
        ),
        dpi=150
    )

    plt.close(fig)

    # Save individual predictions
    prediction_df = pd.DataFrame({
        "true_label": np.array(classes)[true_labels],
        "predicted_label": np.array(classes)[predicted_labels],
        "correct": true_labels == predicted_labels
    })

    for i, class_name in enumerate(classes):
        prediction_df[f"prob_{class_name}"] = predictions[:, i]

    prediction_df.to_csv(
        os.path.join(
            results_dir,
            f"{model_name.lower()}_test_predictions.csv"
        ),
        index_label="index"
    )

    return metrics


# ============================================================
# Training History Plot
# ============================================================

def plot_history(history, model_name, results_dir):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        history.history["accuracy"],
        label="Training"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(
        history.history["loss"],
        label="Training"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Validation"
    )

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(model_name)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            f"{model_name.lower()}_training_history.png"
        ),
        dpi=150
    )

    plt.close(fig)


# ============================================================
# Load and Prepare Data
# ============================================================

def load_data():

    section("Step 1: Load and Preprocess Data")

    train_images, train_labels = preprocess_dataset(
        training_folder_path,
        size=image_size
    )

    test_images, test_labels = preprocess_dataset(
        testing_folder_path,
        size=image_size
    )

    # EfficientNetB0 expects RGB values in the 0-255 range.
    # Do NOT divide by 255.

    x_train_full = train_images.astype(np.float32)
    x_test = test_images.astype(np.float32)

    y_train_full = to_categorical(
        train_labels,
        num_classes=len(classes)
    )

    y_test = to_categorical(
        test_labels,
        num_classes=len(classes)
    )

    # Stratified train-validation split.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        stratify=train_labels,
        random_state=SEED
    )

    del train_images, x_train_full
    gc.collect()

    show(
        "Dataset splits",
        pd.DataFrame({
            "Images": [
                len(x_train),
                len(x_val),
                len(x_test)
            ],
            "Image shape": [
                x_train.shape[1:],
                x_val.shape[1:],
                x_test.shape[1:]
            ],
            "Dtype": [
                x_train.dtype,
                x_val.dtype,
                x_test.dtype
            ],
            "Pixel range": [
                f"{x_train.min():.0f}-{x_train.max():.0f}",
                f"{x_val.min():.0f}-{x_val.max():.0f}",
                f"{x_test.min():.0f}-{x_test.max():.0f}"
            ]
        }, index=["Training", "Validation", "Testing"])
    )

    return (
        (x_train, y_train, x_val, y_val),
        x_test,
        y_test
    )


# ============================================================
# Run Experiments
# ============================================================

def run_experiment():

    check_gpu()

    section("EfficientNetB0 Brain Tumor Classification")

    data, x_test, y_test = load_data()

    results = []
    histories = []

    # --------------------------------------------------------
    # Experiment 1: ImageNet Feature Extraction
    # --------------------------------------------------------

    section("Experiment 1: ImageNet Feature Extraction")

    stage1_model = build_efficientnetb0(
        weights="imagenet",
        name="EfficientNetB0_Stage1_Frozen",
        image_size=image_size
    )

    freeze_base(stage1_model)

    stage1_params = count_trainable(stage1_model)

    print(
        f"Trainable parameters: {stage1_params:,}"
    )

    stage1_model, stage1_history, stage1_time, stage1_best = train_model(
        stage1_model,
        "EfficientNetB0_Stage1_Frozen",
        stage1_learning_rate,
        stage1_epochs,
        data,
        RESULTS_DIR
    )

    plot_history(
        stage1_history,
        "EfficientNetB0_Stage1_Frozen",
        RESULTS_DIR
    )

    # --------------------------------------------------------
    # Experiment 2: Fine-Tuning
    # --------------------------------------------------------

    section("Experiment 2: EfficientNetB0 Fine-Tuning")

    # Continue from the best feature-extraction checkpoint.
    stage2_model = stage1_model

    unfreeze_last_block(stage2_model)

    stage2_params = count_trainable(stage2_model)

    print(
        f"\nTrainable parameters: {stage2_params:,}"
    )

    stage2_model, stage2_history, stage2_time, stage2_best = train_model(
        stage2_model,
        "EfficientNetB0_Stage2_FineTuned",
        stage2_learning_rate,
        stage2_epochs,
        data,
        RESULTS_DIR
    )

    plot_history(
        stage2_history,
        "EfficientNetB0_Stage2_FineTuned",
        RESULTS_DIR
    )

    # --------------------------------------------------------
    # Experiment 3: Training From Scratch
    # --------------------------------------------------------

    section("Experiment 3: EfficientNetB0 From Scratch")

    # Release pretrained models before creating the scratch model.
    del stage1_model, stage2_model
    free_gpu_memory()

    scratch_model = build_efficientnetb0(
        weights=None,
        name="EfficientNetB0_From_Scratch",
        image_size=image_size
    )

    scratch_params = count_trainable(scratch_model)

    print(
        f"Trainable parameters: {scratch_params:,}"
    )

    scratch_model, scratch_history, scratch_time, scratch_best = train_model(
        scratch_model,
        "EfficientNetB0_From_Scratch",
        scratch_learning_rate,
        scratch_epochs,
        data,
        RESULTS_DIR
    )

    plot_history(
        scratch_history,
        "EfficientNetB0_From_Scratch",
        RESULTS_DIR
    )

    # --------------------------------------------------------
    # Final Test Evaluation
    # --------------------------------------------------------

    section("Final Held-Out Test Evaluation")

    experiments = [
        (
            "EfficientNetB0_Stage1_Frozen",
            stage1_history,
            stage1_time,
            stage1_best,
            stage1_params,
            os.path.join(
                RESULTS_DIR,
                "efficientnetb0_stage1_frozen_best.keras"
            )
        ),
        (
            "EfficientNetB0_Stage2_FineTuned",
            stage2_history,
            stage2_time,
            stage2_best,
            stage2_params,
            os.path.join(
                RESULTS_DIR,
                "efficientnetb0_stage2_finetuned_best.keras"
            )
        ),
        (
            "EfficientNetB0_From_Scratch",
            scratch_history,
            scratch_time,
            scratch_best,
            scratch_params,
            os.path.join(
                RESULTS_DIR,
                "efficientnetb0_from_scratch_best.keras"
            )
        )
    ]

    # Reload and evaluate each saved final checkpoint.
    for name, history, duration, best_epoch, params, checkpoint in experiments:

        section(f"Test Evaluation: {name}")

        model = keras.models.load_model(checkpoint)

        metrics = evaluate_test(
            model,
            name,
            x_test,
            y_test,
            RESULTS_DIR
        )

        metrics["Trainable Parameters"] = params
        metrics["Epochs Run"] = len(history.history["loss"])
        metrics["Best Epoch"] = best_epoch
        metrics["Training Time (min)"] = duration

        results.append(metrics)

        del model
        free_gpu_memory()

    # --------------------------------------------------------
    # Save Comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        os.path.join(
            RESULTS_DIR,
            "efficientnetb0_comparison.csv"
        ),
        index=False
    )

    section("EfficientNetB0 Experiment Comparison")

    show(
        "Final comparison",
        comparison.set_index("Model")
    )

    print(
        f"\nAll results saved to:\n{RESULTS_DIR}"
    )


if __name__ == "__main__":
    run_experiment()
