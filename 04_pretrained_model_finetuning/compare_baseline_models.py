
# -*- coding: utf-8 -*-
"""
@author: Abdallah Divkar
Baseline comparison and class-level analysis
COMP263 - Brain Tumor MRI Classification

Architectures:
    - EfficientNetB0
    - DenseNet121

Generates:
    1. Test accuracy comparison
    2. Macro F1 comparison
    3. Training time comparison
    4. Macro precision vs. macro recall
    5. Class-level precision, recall, and F1 comparison
    6. Six confusion matrices
    7. CSV exports of class metrics and confusion matrices

Confusion matrices and class-level metrics are calculated from
the saved test prediction CSV files, not from PNG images.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_ROOT = PROJECT_ROOT / "results"

EFFICIENTNET_DIR = (
    RESULTS_ROOT / "pretrained_efficientnetb0_128"
)

DENSENET_DIR = (
    RESULTS_ROOT / "pretrained_densenet121_128"
)

OUTPUT_DIR = RESULTS_ROOT / "baseline_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Assumes the original integer encoding:
# 0 = glioma
# 1 = meningioma
# 2 = notumor
# 3 = pituitary
#
# If your prediction CSVs use a different encoding,
# update this mapping.

NUMERIC_LABEL_MAP = {
    "0": "glioma",
    "1": "meningioma",
    "2": "notumor",
    "3": "pituitary"
}


# Overall baseline metrics from the six experiments.
# Values are percentages, except training time (minutes).

baseline_data = [
    {
        "Architecture": "EfficientNetB0",
        "Experiment": "Frozen",
        "Accuracy": 80.56,
        "Precision": 80.84,
        "Recall": 80.35,
        "Macro_F1": 79.56,
        "Epochs_Run": 10,
        "Best_Epoch": 10,
        "Training_Time": 5.30
    },
    {
        "Architecture": "EfficientNetB0",
        "Experiment": "Fine-tuned",
        "Accuracy": 83.46,
        "Precision": 83.69,
        "Recall": 83.28,
        "Macro_F1": 82.72,
        "Epochs_Run": 20,
        "Best_Epoch": 18,
        "Training_Time": 12.00
    },
    {
        "Architecture": "EfficientNetB0",
        "Experiment": "Scratch",
        "Accuracy": 92.55,
        "Precision": 92.96,
        "Recall": 92.44,
        "Macro_F1": 92.38,
        "Epochs_Run": 30,
        "Best_Epoch": 28,
        "Training_Time": 24.76
    },
    {
        "Architecture": "DenseNet121",
        "Experiment": "Frozen",
        "Accuracy": 82.95,
        "Precision": 84.22,
        "Recall": 82.77,
        "Macro_F1": 82.71,
        "Epochs_Run": 10,
        "Best_Epoch": 7,
        "Training_Time": 8.15
    },
    {
        "Architecture": "DenseNet121",
        "Experiment": "Fine-tuned",
        "Accuracy": 88.57,
        "Precision": 88.90,
        "Recall": 88.44,
        "Macro_F1": 88.26,
        "Epochs_Run": 20,
        "Best_Epoch": 19,
        "Training_Time": 19.08
    },
    {
        "Architecture": "DenseNet121",
        "Experiment": "Scratch",
        "Accuracy": 88.32,
        "Precision": 89.57,
        "Recall": 88.13,
        "Macro_F1": 87.82,
        "Epochs_Run": 21,
        "Best_Epoch": 16,
        "Training_Time": 51.49
    }
]

baseline_df = pd.DataFrame(baseline_data)

baseline_df["Model"] = (
    baseline_df["Architecture"]
    + " - "
    + baseline_df["Experiment"]
)


# ============================================================
# 3. GENERAL PLOTTING FUNCTION
# ============================================================

def create_bar_chart(
    data,
    values_column,
    title,
    ylabel,
    filename,
    ylim=None
):
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        data["Model"],
        data[values_column]
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Architecture and training strategy")
    ax.set_ylabel(ylabel)

    ax.tick_params(
        axis="x",
        labelrotation=20
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 4. OVERALL BASELINE VISUALIZATIONS
# ============================================================

'''def generate_overall_visualizations():

    # Test accuracy
    create_bar_chart(
        baseline_df,
        "Accuracy",
        "Test Accuracy Across Six Baseline Experiments",
        "Test Accuracy (%)",
        "test_accuracy_comparison.png",
        (0, 105)
    )

    # Macro F1
    create_bar_chart(
        baseline_df,
        "Macro_F1",
        "Macro F1 Across Six Baseline Experiments",
        "Macro F1 (%)",
        "macro_f1_comparison.png",
        (0, 105)
    )

    # Training time
    create_bar_chart(
        baseline_df,
        "Training_Time",
        "Training Time Across Six Baseline Experiments",
        "Training Time (minutes)",
        "training_time_comparison.png"
    )

    # Macro precision vs. macro recall
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in baseline_df.iterrows():

        ax.scatter(
            row["Recall"],
            row["Precision"],
            s=110
        )

        ax.annotate(
            row["Model"],
            (
                row["Recall"],
                row["Precision"]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_title(
        "Macro Precision vs. Macro Recall",
        fontsize=14
    )

    ax.set_xlabel("Macro Recall (%)")
    ax.set_ylabel("Macro Precision (%)")

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "macro_precision_vs_recall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # Save overall comparison table
    baseline_df.to_csv(
        OUTPUT_DIR / "baseline_comparison.csv",
        index=False
    )

    print("Overall baseline visualizations generated.")'''


# ============================================================
# 5. PREDICTION CSV CONFIGURATION
# ============================================================

# These are the prediction CSV filenames generated by
# the six experiments.

prediction_files = {
    "EfficientNetB0 - Frozen":
        EFFICIENTNET_DIR /
        "efficientnetb0_stage1_frozen_test_predictions.csv",

    "EfficientNetB0 - Fine-tuned":
        EFFICIENTNET_DIR /
        "efficientnetb0_stage2_finetuned_test_predictions.csv",

    "EfficientNetB0 - Scratch":
        EFFICIENTNET_DIR /
        "efficientnetb0_from_scratch_test_predictions.csv",

    "DenseNet121 - Frozen":
        DENSENET_DIR /
        "densenet121_stage1_frozen_test_predictions.csv",

    "DenseNet121 - Fine-tuned":
        DENSENET_DIR /
        "densenet121_stage2_finetuned_test_predictions.csv",

    "DenseNet121 - Scratch":
        DENSENET_DIR /
        "densenet121_from_scratch_test_predictions.csv"
}


# ============================================================
# 6. LABEL NORMALIZATION
# ============================================================

def normalize_label(value):

    value = str(value).strip().lower()

    # Handle numeric class labels
    if value in NUMERIC_LABEL_MAP:
        return NUMERIC_LABEL_MAP[value]

    # Normalize common naming variations
    value = re.sub(r"[\s_-]+", "", value)

    aliases = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "notumor": "notumor",
        "none": "notumor",
        "pituitary": "pituitary"
    }

    if value in aliases:
        return aliases[value]

    raise ValueError(
        f"Unrecognized class label: {value}. "
        "Check your prediction CSV labels and "
        "update normalize_label() if necessary."
    )


# ============================================================
# 7. LOAD TEST PREDICTIONS
# ============================================================

TRUE_COLUMN_CANDIDATES = [
    "true_label",
    "actual_label",
    "actual",
    "y_true",
    "true",
    "label",
    "ground_truth"
]

PRED_COLUMN_CANDIDATES = [
    "predicted_label",
    "predicted",
    "prediction",
    "y_pred",
    "pred_label",
    "pred"
]


def find_column(df, candidates, description):

    lower_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate in lower_columns:
            return lower_columns[candidate]

    raise ValueError(
        f"Could not identify the {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {candidates}"
    )


def load_predictions(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    true_column = find_column(
        df,
        TRUE_COLUMN_CANDIDATES,
        "true-label"
    )

    pred_column = find_column(
        df,
        PRED_COLUMN_CANDIDATES,
        "predicted-label"
    )

    y_true = df[true_column].apply(normalize_label)
    y_pred = df[pred_column].apply(normalize_label)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction length mismatch in {csv_path}"
        )

    return y_true, y_pred


# ============================================================
# 8. CLASS-LEVEL METRICS
# ============================================================

def analyze_class_performance():

    all_class_metrics = []
    all_confusion_matrices = {}

    for model_name, csv_path in prediction_files.items():

        print("\n" + "=" * 65)
        print(model_name)
        print("=" * 65)

        y_true, y_pred = load_predictions(csv_path)

        report = classification_report(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        print(
            classification_report(
                y_true,
                y_pred,
                labels=CLASS_NAMES,
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

        # Store per-class metrics
        for class_name in CLASS_NAMES:

            metrics = report[class_name]

            all_class_metrics.append({
                "Model": model_name,
                "Class": class_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1_Score": metrics["f1-score"],
                "Support": metrics["support"]
            })

        # Calculate confusion matrix
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_NAMES
        )

        all_confusion_matrices[model_name] = cm

    class_metrics_df = pd.DataFrame(all_class_metrics)

    class_metrics_df.to_csv(
        OUTPUT_DIR / "class_level_metrics.csv",
        index=False
    )

    return class_metrics_df, all_confusion_matrices


# ============================================================
# 9. CLASS-LEVEL METRIC VISUALIZATIONS
# ============================================================

def plot_class_metrics(class_metrics_df):

    metrics_to_plot = [
        ("Precision", "Class-Level Precision"),
        ("Recall", "Class-Level Recall"),
        ("F1_Score", "Class-Level F1 Score")
    ]

    for metric_column, title in metrics_to_plot:

        fig, ax = plt.subplots(figsize=(13, 7))

        pivot_df = class_metrics_df.pivot(
            index="Class",
            columns="Model",
            values=metric_column
        )

        pivot_df.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Tumor Class")
        ax.set_ylabel(metric_column)
        ax.set_ylim(0, 1.05)

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.legend(
            title="Experiment",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8
        )

        fig.tight_layout()

        filename = (
            metric_column.lower()
            + "_class_comparison.png"
        )

        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)


# ============================================================
# 10. CONFUSION MATRIX COMPARISON
# ============================================================

def plot_confusion_matrices(confusion_matrices):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 11)
    )

    axes = axes.flatten()

    for ax, (model_name, cm) in zip(
        axes,
        confusion_matrices.items()
    ):

        image = ax.imshow(
            cm,
            interpolation="nearest"
        )

        ax.set_title(model_name, fontsize=10)

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))

        ax.set_xticklabels(
            CLASS_NAMES,
            rotation=35,
            ha="right",
            fontsize=8
        )

        ax.set_yticklabels(
            CLASS_NAMES,
            fontsize=8
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Annotate cell counts
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9
                )

    fig.colorbar(
        image,
        ax=axes.tolist(),
        shrink=0.75,
        label="Number of Images"
    )

    fig.suptitle(
        "Confusion Matrix Comparison - Six Baseline Experiments",
        fontsize=15
    )

    fig.savefig(
        OUTPUT_DIR / "six_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # Export individual confusion matrices as CSV
    for model_name, cm in confusion_matrices.items():

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "")
        )

        cm_df = pd.DataFrame(
            cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        )

        cm_df.index.name = "Actual / Predicted"

        cm_df.to_csv(
            OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
        )


# ============================================================
# 11. GLIOMA RECALL COMPARISON
# ============================================================

def plot_glioma_recall(class_metrics_df):

    glioma_df = class_metrics_df[
        class_metrics_df["Class"] == "glioma"
    ].copy()

    glioma_df = glioma_df.sort_values(
        "Recall",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        glioma_df["Model"],
        glioma_df["Recall"] * 100
    )

    ax.set_title(
        "Glioma Recall Across Six Baseline Experiments",
        fontsize=14
    )

    ax.set_ylabel("Glioma Recall (%)")
    ax.set_xlabel("Experiment")
    ax.set_ylim(0, 105)

    ax.tick_params(
        axis="x",
        labelrotation=20
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.bar_label(
        bars,
        fmt="%.2f%%",
        padding=3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "glioma_recall_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    print("\nStarting baseline comparison analysis...\n")

    # Overall metrics and visualizations
    generate_overall_visualizations()

    # Class-level metrics from test predictions
    class_metrics_df, confusion_matrices = (
        analyze_class_performance()
    )

    # Class-level plots
    plot_class_metrics(class_metrics_df)

    # Six confusion matrices
    plot_confusion_matrices(confusion_matrices)

    # Glioma recall
    plot_glioma_recall(class_metrics_df)

    print("\n" + "=" * 65)
    print("BASELINE ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"\nAll outputs saved to:\n{OUTPUT_DIR}")