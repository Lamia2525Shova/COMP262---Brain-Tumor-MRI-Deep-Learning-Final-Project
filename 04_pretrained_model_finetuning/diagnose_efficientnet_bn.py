
# -*- coding: utf-8 -*-
"""
@author: Abdallah Divkar

EfficientNetB0 Batch Normalization diagnostic.

Purpose:
    Reproduce the EfficientNetB0 scratch-training configuration
    and record Batch Normalization moving statistics after every epoch.

Important:
    - Does not modify the original baseline script.
    - Does not load or evaluate the held-out test dataset.
    - Saves outputs in a separate diagnostic directory.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

import gc
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from sklearn.model_selection import train_test_split

# Import the existing project implementation.
# Importing this module does not execute its run_experiment()
# because that function is protected by its main guard.
import pretrained_efficientnetb0_128 as baseline


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = baseline.image_size
BATCH_SIZE = baseline.batch_size
EPOCHS = baseline.scratch_epochs
LEARNING_RATE = baseline.scratch_learning_rate
PATIENCE = baseline.patience
SEED = baseline.SEED

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "efficientnet_bn_diagnostic"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Batch Normalization Monitoring Callback
# ============================================================

class BatchNormMonitor(keras.callbacks.Callback):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.bn_layers = [
            layer
            for layer in backbone.layers
            if isinstance(
                layer,
                keras.layers.BatchNormalization
            )
        ]

        self.layer_records = []
        self.epoch_records = []

        print(
            f"\nMonitoring {len(self.bn_layers)} "
            "Batch Normalization layers."
        )

    def on_epoch_end(self, epoch, logs=None):

        all_means = []
        all_variances = []

        for layer in self.bn_layers:

            moving_mean = np.asarray(
                keras.ops.convert_to_numpy(
                    layer.moving_mean
                )
            )

            moving_variance = np.asarray(
                keras.ops.convert_to_numpy(
                    layer.moving_variance
                )
            )

            all_means.append(moving_mean.flatten())
            all_variances.append(moving_variance.flatten())

            self.layer_records.append({
                "epoch": epoch + 1,
                "layer": layer.name,
                "moving_mean_avg": float(
                    np.mean(moving_mean)
                ),
                "moving_mean_std": float(
                    np.std(moving_mean)
                ),
                "moving_variance_avg": float(
                    np.mean(moving_variance)
                ),
                "moving_variance_std": float(
                    np.std(moving_variance)
                ),
                "moving_variance_min": float(
                    np.min(moving_variance)
                ),
                "moving_variance_max": float(
                    np.max(moving_variance)
                )
            })

        combined_means = np.concatenate(all_means)
        combined_variances = np.concatenate(all_variances)

        self.epoch_records.append({
            "epoch": epoch + 1,
            "moving_mean_abs_avg": float(
                np.mean(np.abs(combined_means))
            ),
            "moving_mean_std": float(
                np.std(combined_means)
            ),
            "moving_variance_avg": float(
                np.mean(combined_variances)
            ),
            "moving_variance_std": float(
                np.std(combined_variances)
            ),
            "moving_variance_min": float(
                np.min(combined_variances)
            ),
            "moving_variance_max": float(
                np.max(combined_variances)
            )
        })

        print(
            "\nBN diagnostic | "
            f"Epoch {epoch + 1} | "
            f"Mean abs: {np.mean(np.abs(combined_means)):.5f} | "
            f"Variance avg: {np.mean(combined_variances):.5f} | "
            f"Variance range: "
            f"{np.min(combined_variances):.5f} - "
            f"{np.max(combined_variances):.5f}"
        )


# ============================================================
# Load Training Data Only
# ============================================================

def load_training_data():

    print("\nLoading and preprocessing training data...")

    train_images, train_labels = baseline.preprocess_dataset(
        baseline.training_folder_path,
        size=IMAGE_SIZE
    )

    x_train_full = train_images.astype(np.float32)

    y_train_full = baseline.to_categorical(
        train_labels,
        num_classes=len(baseline.classes)
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        stratify=train_labels,
        random_state=SEED
    )

    print("\nDataset split:")
    print("Training:", x_train.shape)
    print("Validation:", x_val.shape)
    print("Training dtype:", x_train.dtype)
    print(
        "Pixel range:",
        float(x_train.min()),
        "to",
        float(x_train.max())
    )

    del train_images, x_train_full
    gc.collect()

    return x_train, y_train, x_val, y_val


# ============================================================
# Plot Diagnostic Results
# ============================================================

def save_plots(history, bn_summary):

    history_df = pd.DataFrame(history.history)

    # Training and validation curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        history_df["loss"],
        label="Training Loss"
    )
    axes[0].plot(
        history_df["val_loss"],
        label="Validation Loss"
    )
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history_df["accuracy"],
        label="Training Accuracy"
    )
    axes[1].plot(
        history_df["val_accuracy"],
        label="Validation Accuracy"
    )
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "diagnostic_training_history.png"
        ),
        dpi=300
    )

    plt.close()

    # Batch Normalization statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        bn_summary["epoch"],
        bn_summary["moving_mean_abs_avg"],
        marker="o"
    )
    axes[0].set_title("BN Moving Mean: Average Absolute Value")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average Absolute Value")
    axes[0].grid(True)

    axes[1].plot(
        bn_summary["epoch"],
        bn_summary["moving_variance_avg"],
        marker="o",
        label="Average"
    )

    axes[1].plot(
        bn_summary["epoch"],
        bn_summary["moving_variance_min"],
        linestyle="--",
        label="Minimum"
    )

    axes[1].plot(
        bn_summary["epoch"],
        bn_summary["moving_variance_max"],
        linestyle="--",
        label="Maximum"
    )

    axes[1].set_title("BN Moving Variance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Variance")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "batch_normalization_statistics.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# Run Diagnostic
# ============================================================

def run_diagnostic():

    print("=" * 65)
    print("EfficientNetB0 Batch Normalization Diagnostic")
    print("=" * 65)

    baseline.check_gpu()

    # Recreate the original training/validation split.
    x_train, y_train, x_val, y_val = load_training_data()

    # Reset seed before model initialization.
    keras.utils.set_random_seed(SEED)

    # Reuse the original model-building implementation.
    model = baseline.build_efficientnetb0(
        weights=None,
        name="EfficientNetB0_BN_Diagnostic",
        image_size=IMAGE_SIZE
    )

    backbone = model.get_layer("efficientnetb0")

    bn_layers = [
        layer
        for layer in backbone.layers
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        )
    ]

    print("\nModel configuration:")
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("BN layers:", len(bn_layers))

    print(
        "Trainable parameters:",
        sum(
            int(np.prod(keras.ops.convert_to_numpy(w).shape))
            for w in model.trainable_weights
        )
    )

    # Monitor BN statistics after every epoch.
    bn_monitor = BatchNormMonitor(backbone)

    # Preserve original optimizer and early-stopping settings.
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        "efficientnet_bn_diagnostic_best.keras"
    )

    callbacks = [
        bn_monitor,

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
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

    print("\nStarting diagnostic training...")

    start_time = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=baseline.balanced_class_weights(y_train),
        callbacks=callbacks,
        verbose=2
    )

    duration = (time.perf_counter() - start_time) / 60

    # Save epoch-by-epoch training history.
    history_df = pd.DataFrame(history.history)
    history_df.insert(
        0,
        "epoch",
        range(1, len(history_df) + 1)
    )

    history_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "training_history.csv"
        ),
        index=False
    )

    # Save per-layer BN statistics.
    layer_df = pd.DataFrame(bn_monitor.layer_records)

    layer_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "batch_norm_layer_statistics.csv"
        ),
        index=False
    )

    # Save overall BN statistics.
    summary_df = pd.DataFrame(bn_monitor.epoch_records)

    summary_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "batch_norm_epoch_summary.csv"
        ),
        index=False
    )

    save_plots(history, summary_df)

    print("\n" + "=" * 65)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 65)

    print(f"Training duration: {duration:.2f} minutes")
    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.6f}")

    print("\nResults saved to:")
    print(RESULTS_DIR)

    print("\nNo held-out test evaluation was performed.")

    del model
    gc.collect()
    baseline.free_gpu_memory()


if __name__ == "__main__":
    run_diagnostic()
