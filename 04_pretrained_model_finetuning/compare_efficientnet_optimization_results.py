r"""
@author: Abdallah Divkar

Consolidated EfficientNetB0 optimization comparison.

This is an analysis-only script. It does NOT load datasets, load model
checkpoints, train models, or rerun evaluation. It uses the recorded
validation and test metrics from the completed experiments.

Run from the project root:
    python .\04_pretrained_model_finetuning\compare_efficientnet_optimization_results.py

Outputs:
    04_pretrained_model_finetuning/results/efficientnet_optimization_comparison/
        efficientnet_optimization_comparison.csv
        validation_accuracy_comparison.png
        validation_loss_comparison.png
        test_metrics_comparison.png
        training_time_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "efficientnet_optimization_comparison"
)

# Values below are transcribed from the completed experiment records.
# None means the metric was intentionally not measured / not available.
EXPERIMENTS = [
    {
        "configuration": "Original Scratch",
        "bn_configuration": "All 49 trainable",
        "learning_rate_strategy": "Adam, fixed 0.001",
        "epochs_run": 30,
        "best_val_accuracy": 0.9641,
        "best_val_accuracy_epoch": 28,
        "best_val_loss": 0.1181,
        "best_val_loss_epoch": 28,
        "test_accuracy": 0.925505,
        "test_macro_precision": 0.929585,
        "test_macro_recall": 0.924378,
        "test_macro_f1": 0.923787,
        "training_time_minutes": 24.76,
        "test_evaluated": True,
        "checkpoint_selection": "Original baseline checkpoint",
    },
    {
        "configuration": "Frozen BN",
        "bn_configuration": "All 49 frozen",
        "learning_rate_strategy": "Adam, fixed 0.001",
        "epochs_run": 30,
        "best_val_accuracy": 0.913444,
        "best_val_accuracy_epoch": 26,
        "best_val_loss": 0.249776,
        "best_val_loss_epoch": 29,
        "test_accuracy": 0.873737,
        "test_macro_precision": 0.875817,
        "test_macro_recall": 0.872241,
        "test_macro_f1": 0.869477,
        "training_time_minutes": 18.416,
        "test_evaluated": True,
        "checkpoint_selection": "Best validation-loss checkpoint",
    },
    {
        "configuration": "Selective BN",
        "bn_configuration": "4 selected frozen; 45 trainable",
        "learning_rate_strategy": "Adam, fixed 0.001",
        "epochs_run": 22,
        "best_val_accuracy": 0.918969,
        "best_val_accuracy_epoch": 17,
        "best_val_loss": 0.261883,
        "best_val_loss_epoch": 17,
        "test_accuracy": None,
        "test_macro_precision": None,
        "test_macro_recall": None,
        "test_macro_f1": None,
        "training_time_minutes": 18.43,
        "test_evaluated": False,
        "checkpoint_selection": "Validation diagnostic only; no test evaluation",
    },
    {
        "configuration": "ReduceLROnPlateau",
        "bn_configuration": "All 49 trainable",
        "learning_rate_strategy": "ReduceLROnPlateau",
        "epochs_run": 30,
        "best_val_accuracy": 0.959484,
        "best_val_accuracy_epoch": 29,
        "best_val_loss": 0.141863,
        "best_val_loss_epoch": 25,
        "test_accuracy": 0.9078,
        "test_macro_precision": 0.9131,
        "test_macro_recall": 0.9065,
        "test_macro_f1": 0.9054,
        "training_time_minutes": 23.05,
        "test_evaluated": True,
        "checkpoint_selection": "Best validation-loss checkpoint (epoch 25)",
    },
]


def save_validation_accuracy_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    values = df["best_val_accuracy"] * 100
    bars = ax.bar(df["configuration"], values)
    ax.set_title("EfficientNetB0 Optimization: Best Validation Accuracy")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def save_validation_loss_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["configuration"], df["best_val_loss"])
    ax.set_title("EfficientNetB0 Optimization: Best Validation Loss")
    ax.set_ylabel("Validation loss (lower is better)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, df["best_val_loss"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_loss_comparison.png", dpi=300)
    plt.close(fig)


def save_test_metrics_plot(df: pd.DataFrame) -> None:
    metric_columns = {
        "Accuracy": "test_accuracy",
        "Macro precision": "test_macro_precision",
        "Macro recall": "test_macro_recall",
        "Macro F1": "test_macro_f1",
    }
    evaluated = df[df["test_evaluated"]].copy()
    x = range(len(evaluated))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, (label, column) in enumerate(metric_columns.items()):
        positions = [position + (index - 1.5) * width for position in x]
        values = evaluated[column] * 100
        bars = ax.bar(positions, values, width, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("EfficientNetB0 Optimization: Held-Out Test Metrics")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(list(x))
    ax.set_xticklabels(evaluated["configuration"], rotation=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_metrics_comparison.png", dpi=300)
    plt.close(fig)


def save_training_time_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["configuration"], df["training_time_minutes"])
    ax.set_title("EfficientNetB0 Optimization: Training Time")
    ax.set_ylabel("Training time (minutes)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, df["training_time_minutes"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_time_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(EXPERIMENTS)

    csv_path = OUTPUT_DIR / "efficientnet_optimization_comparison.csv"
    df.to_csv(csv_path, index=False)

    save_validation_accuracy_plot(df)
    save_validation_loss_plot(df)
    save_test_metrics_plot(df)
    save_training_time_plot(df)

    display_columns = [
        "configuration",
        "best_val_accuracy",
        "best_val_accuracy_epoch",
        "best_val_loss",
        "best_val_loss_epoch",
        "test_accuracy",
        "test_macro_f1",
        "training_time_minutes",
        "test_evaluated",
    ]
    print("\nEfficientNetB0 optimization comparison")
    print(df[display_columns].to_string(index=False))
    print(f"\nSaved CSV and plots to:\n{OUTPUT_DIR}")
    print("\nNo training, checkpoint loading, or test evaluation was performed.")


if __name__ == "__main__":
    main()
