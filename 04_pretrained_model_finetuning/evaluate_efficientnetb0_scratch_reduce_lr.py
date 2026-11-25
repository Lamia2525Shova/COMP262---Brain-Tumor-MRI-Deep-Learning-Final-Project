
# -*- coding: utf-8 -*-
"""
@author: Abdallah Divkar

Evaluate the EfficientNetB0 Scratch + ReduceLROnPlateau checkpoint.

This is an evaluation-only script:
- Does not train or retrain the model.
- Does not modify the original EfficientNetB0 baseline.
- Loads the saved best-validation-loss checkpoint.
- Evaluates only on the held-out test dataset.
- Saves metrics, classification report, confusion matrix,
  and individual predictions in the experiment results directory.
"""

import os
import sys
import gc

# ============================================================
# Environment and Project Setup
# ============================================================

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_here)

sys.path[:0] = [
    os.path.join(PROJECT_ROOT, "01_data_exploration_preprocessing"),
    os.path.join(PROJECT_ROOT, "utils")
]

os.chdir(PROJECT_ROOT)

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_exploration import (
    classes,
    section,
    training_folder_path,
    testing_folder_path
)

from preprocessing import preprocess_dataset


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

MODEL_NAME = "EfficientNetB0_Scratch_ReduceLR"

RESULTS_DIR = os.path.join(
    _here,
    "results",
    "efficientnetb0_scratch_reduce_lr"
)

CHECKPOINT_PATH = os.path.join(
    RESULTS_DIR,
    "efficientnetb0_scratch_reduce_lr_best.keras"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Hardware Verification
# ============================================================

def check_environment():

    print("\nEnvironment verification")

    print("Keras:", keras.__version__)
    print("Keras backend:", keras.backend.backend())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if keras.backend.backend() != "torch":
        raise RuntimeError(
            "Expected Keras PyTorch backend."
        )

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU unavailable. Evaluation will use CPU.")


# ============================================================
# Load Saved Checkpoint
# ============================================================

def load_checkpoint():

    section("Load ReduceLROnPlateau Checkpoint")

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print(f"Checkpoint: {CHECKPOINT_PATH}")

    model = keras.models.load_model(
        CHECKPOINT_PATH,
        compile=False
    )

    print("\nCheckpoint loaded successfully.")
    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    expected_output_units = len(classes)

    if model.output_shape[-1] != expected_output_units:
        raise ValueError(
            f"Expected {expected_output_units} output classes, "
            f"but checkpoint has {model.output_shape[-1]}."
        )

    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    return model


# ============================================================
# Load and Preprocess Test Data Only
# ============================================================

def load_test_data():

    section("Load and Preprocess Held-Out Test Data")

    # Deliberately load only the original testing directory.
    # No training or validation data is loaded.

    test_images, test_labels = preprocess_dataset(
        testing_folder_path,
        size=IMAGE_SIZE
    )

    x_test = test_images.astype(np.float32)

    y_test = np.asarray(test_labels, dtype=np.int64)

    # EfficientNetB0 uses internal input rescaling.
    # Preserve RGB pixel values in the 0-255 range.
    # Do NOT divide by 255.

    if x_test.ndim != 4 or x_test.shape[1:] != (
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ):
        raise ValueError(
            f"Unexpected test image shape: {x_test.shape}"
        )

    if len(x_test) != len(y_test):
        raise ValueError(
            "Test image and label counts do not match."
        )

    if not np.all(np.isfinite(x_test)):
        raise ValueError(
            "Test images contain NaN or infinite values."
        )

    if np.any(y_test < 0) or np.any(y_test >= len(classes)):
        raise ValueError(
            "Test labels contain invalid class indices."
        )

    print("\nTest dataset:")
    print(f"Images:      {len(x_test)}")
    print(f"Image shape: {x_test.shape[1:]}")
    print(f"Data type:   {x_test.dtype}")
    print(
        f"Pixel range: {x_test.min():.0f}–{x_test.max():.0f}"
    )

    print("\nClass order:")
    for index, class_name in enumerate(classes):
        print(f"{index}: {class_name}")

    print("\nInput preprocessing:")
    print("RGB images, float32, pixel range 0–255.")
    print("No division by 255.")

    return x_test, y_test


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(model, x_test, y_test):

    section("Generate Test Predictions")

    predictions = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    if predictions.shape != (len(y_test), len(classes)):
        raise ValueError(
            f"Unexpected prediction shape: {predictions.shape}"
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Model predictions contain NaN or infinite values."
        )

    true_labels = y_test
    predicted_labels = np.argmax(predictions, axis=1)

    # --------------------------------------------------------
    # Aggregate Metrics
    # --------------------------------------------------------

    metrics = {
        "Model": MODEL_NAME,
        "Test Images": len(y_test),

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

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)

    for key, value in metrics.items():

        if key == "Model":
            print(f"{key}: {value}")

        elif key == "Test Images":
            print(f"{key}: {value}")

        else:
            print(f"{key}: {value:.4f} ({value * 100:.2f}%)")

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(
        RESULTS_DIR,
        "efficientnetb0_scratch_reduce_lr_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(f"Classification report saved: {report_path}")

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(len(classes))
    )

    print("\nConfusion Matrix:")
    print(matrix)

    matrix_path = os.path.join(
        RESULTS_DIR,
        "efficientnetb0_scratch_reduce_lr_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    ).to_csv(matrix_path)

    fig, ax = plt.subplots(figsize=(8, 6))

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
        title=f"{MODEL_NAME} - Test Confusion Matrix"
    )

    plt.tight_layout()

    figure_path = os.path.join(
        RESULTS_DIR,
        "efficientnetb0_scratch_reduce_lr_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Confusion matrix CSV saved: {matrix_path}")
    print(f"Confusion matrix plot saved: {figure_path}")

    # --------------------------------------------------------
    # Individual Predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "true_label": np.array(classes)[true_labels],
        "predicted_label": np.array(classes)[predicted_labels],
        "correct": true_labels == predicted_labels
    })

    for index, class_name in enumerate(classes):

        prediction_df[f"prob_{class_name}"] = predictions[:, index]

    prediction_path = os.path.join(
        RESULTS_DIR,
        "efficientnetb0_scratch_reduce_lr_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index_label="index"
    )

    print(f"Individual predictions saved: {prediction_path}")

    # --------------------------------------------------------
    # Save Aggregate Metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        RESULTS_DIR,
        "efficientnetb0_scratch_reduce_lr_final_test_evaluation.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False
    )

    print(f"Aggregate metrics saved: {metrics_path}")

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    check_environment()

    section("EfficientNetB0 Scratch - ReduceLROnPlateau Test Evaluation")

    print("\nEvaluation configuration:")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Results directory: {RESULTS_DIR}")

    model = load_checkpoint()

    x_test, y_test = load_test_data()

    metrics = evaluate_model(
        model,
        x_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(f"\nAll evaluation outputs saved to:\n{RESULTS_DIR}")

    del model, x_test, y_test
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    main()
