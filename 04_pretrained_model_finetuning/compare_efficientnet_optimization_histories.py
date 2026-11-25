'''
@author: Abdallah Divkar
'''
import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "Frozen BN": os.path.join(
        BASE_DIR,
        "results",
        "efficientnetb0_scratch_frozen_bn",
        "training_history.csv"
    ),

    "Selective BN": os.path.join(
        BASE_DIR,
        "results",
        "efficientnetb0_scratch_selective_bn",
        "training_history.csv"
    ),

    "ReduceLROnPlateau": os.path.join(
        BASE_DIR,
        "results",
        "efficientnetb0_scratch_reduce_lr",
        "training_history.csv"
    )
}

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "efficientnet_optimization_history_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load and Validate Histories
# ============================================================

def load_histories():

    histories = {}

    required_columns = {
        "epoch",
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss"
    }

    for experiment_name, file_path in EXPERIMENTS.items():

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing history file for {experiment_name}:\n"
                f"{file_path}"
            )

        df = pd.read_csv(file_path)

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{experiment_name} is missing columns: {missing}"
            )

        histories[experiment_name] = df

        print(
            f"{experiment_name}: "
            f"{len(df)} epochs loaded"
        )

    return histories


# ============================================================
# Generate Numerical Comparison
# ============================================================

def create_summary(histories):

    rows = []

    for experiment_name, df in histories.items():

        best_loss_index = df["val_loss"].idxmin()
        best_accuracy_index = df["val_accuracy"].idxmax()

        rows.append({
            "Experiment": experiment_name,
            "Epochs Completed": len(df),

            "Best Validation Accuracy": float(
                df.loc[best_accuracy_index, "val_accuracy"]
            ),

            "Best Accuracy Epoch": int(
                df.loc[best_accuracy_index, "epoch"]
            ),

            "Best Validation Loss": float(
                df.loc[best_loss_index, "val_loss"]
            ),

            "Best Loss Epoch": int(
                df.loc[best_loss_index, "epoch"]
            ),

            "Accuracy At Best Loss Epoch": float(
                df.loc[best_loss_index, "val_accuracy"]
            ),

            "Final Training Accuracy": float(
                df.iloc[-1]["accuracy"]
            ),

            "Final Validation Accuracy": float(
                df.iloc[-1]["val_accuracy"]
            ),

            "Final Training Loss": float(
                df.iloc[-1]["loss"]
            ),

            "Final Validation Loss": float(
                df.iloc[-1]["val_loss"]
            )
        })

    summary_df = pd.DataFrame(rows)

    output_path = os.path.join(
        OUTPUT_DIR,
        "optimization_history_summary.csv"
    )

    summary_df.to_csv(output_path, index=False)

    print("\nNumerical comparison:")
    print(summary_df.to_string(index=False))

    print(f"\nSummary saved: {output_path}")

    return summary_df


# ============================================================
# Plot Validation Accuracy
# ============================================================

def plot_validation_accuracy(histories):

    plt.figure(figsize=(11, 6))

    for name, df in histories.items():

        plt.plot(
            df["epoch"],
            df["val_accuracy"],
            marker="o",
            markersize=3,
            label=name
        )

    plt.title("EfficientNetB0 Optimization - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "validation_accuracy_comparison.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Plot Validation Loss
# ============================================================

def plot_validation_loss(histories):

    plt.figure(figsize=(11, 6))

    for name, df in histories.items():

        plt.plot(
            df["epoch"],
            df["val_loss"],
            marker="o",
            markersize=3,
            label=name
        )

    plt.title("EfficientNetB0 Optimization - Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "validation_loss_comparison.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Plot Training vs Validation Loss
# ============================================================

def plot_loss_curves(histories):

    fig, axes = plt.subplots(
        len(histories),
        1,
        figsize=(11, 5 * len(histories))
    )

    if len(histories) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, histories.items()):

        ax.plot(
            df["epoch"],
            df["loss"],
            label="Training loss"
        )

        ax.plot(
            df["epoch"],
            df["val_loss"],
            label="Validation loss"
        )

        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "EfficientNetB0 - Training vs Validation Loss",
        fontsize=14
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "training_validation_loss_comparison.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Plot ReduceLROnPlateau Learning Rate
# ============================================================

def plot_learning_rate():

    file_path = os.path.join(
        BASE_DIR,
        "results",
        "efficientnetb0_scratch_reduce_lr",
        "learning_rate_history.csv"
    )

    if not os.path.exists(file_path):
        print("Learning-rate history file not found. Skipping.")
        return

    df = pd.read_csv(file_path)

    plt.figure(figsize=(10, 5))

    plt.step(
        df["epoch"],
        df["learning_rate"],
        where="post",
        marker="o"
    )

    plt.title("EfficientNetB0 - ReduceLROnPlateau Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "reduce_lr_learning_rate.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("\nLoading optimization histories...")

    histories = load_histories()

    create_summary(histories)

    plot_validation_accuracy(histories)

    plot_validation_loss(histories)

    plot_loss_curves(histories)

    plot_learning_rate()

    print("\nComparison completed.")
    print(f"Output directory:\n{OUTPUT_DIR}")

    print(
        "\nNote: Original scratch baseline is excluded from "
        "numerical plots because its original CSV history "
        "was not saved."
    )


if __name__ == "__main__":
    main()