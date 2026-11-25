
"""
EfficientNetB0 from scratch with selectively frozen BatchNormalization layers.

Purpose:
Investigate whether freezing selected deeper BatchNormalization layers
improves validation stability while allowing the remaining BN layers
and convolutional layers to learn.

This experiment:
- Uses random initialization (weights=None).
- Freezes four selected BatchNormalization layers.
- Keeps all convolutional layers trainable.
- Keeps all other BatchNormalization layers trainable.
- Uses validation data for early stopping and checkpoint selection.
- Does NOT load or evaluate the test dataset.
- Saves outputs to a separate experiment directory.
"""

import os
import time
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pretrained_efficientnetb0_128 as baseline


# ============================================================
# Configuration
# ============================================================

keras = baseline.keras

SEED = baseline.SEED
IMAGE_SIZE = baseline.image_size
BATCH_SIZE = baseline.batch_size

MAX_EPOCHS = baseline.scratch_epochs
LEARNING_RATE = baseline.scratch_learning_rate
PATIENCE = baseline.patience

MODEL_NAME = "EfficientNetB0_Scratch_Selective_BN"

SELECTED_BN_LAYERS = {
    "block6b_project_bn",
    "block6c_project_bn",
    "block6d_project_bn",
    "top_bn"
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "efficientnetb0_scratch_selective_bn"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Build Model
# ============================================================

def build_model():

    base = baseline.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Keep the backbone trainable overall.
    base.trainable = True

    # Freeze only the four selected BN layers.
    for layer in base.layers:

        if isinstance(layer, keras.layers.BatchNormalization):

            layer.trainable = (
                layer.name not in SELECTED_BN_LAYERS
            )

        else:
            layer.trainable = True

    # Verify that all selected layer names exist.
    found_bn_names = {
        layer.name
        for layer in base.layers
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        )
    }

    missing_layers = SELECTED_BN_LAYERS - found_bn_names

    if missing_layers:
        raise RuntimeError(
            f"Selected BN layers not found: {sorted(missing_layers)}"
        )

    inputs = keras.layers.Input(
        shape=(*IMAGE_SIZE, 3)
    )

    x = baseline.data_augmentation()(inputs)

    # Do not force the entire backbone into inference mode.
    # Only explicitly frozen BN layers should use inference behavior.
    x = base(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(
        len(baseline.classes),
        activation="softmax"
    )(x)

    model = keras.models.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME
    )

    return model, base


# ============================================================
# BatchNormalization Statistics
# ============================================================

def get_bn_layers(base):

    return [
        layer
        for layer in base.layers
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        )
    ]


def capture_bn_statistics(base):

    statistics = {}

    for layer in get_bn_layers(base):

        statistics[layer.name] = {
            "moving_mean": layer.moving_mean.numpy().copy(),
            "moving_variance": layer.moving_variance.numpy().copy()
        }

    return statistics


def summarize_bn_statistics(statistics, stage):

    rows = []

    for layer_name, values in statistics.items():

        moving_mean = values["moving_mean"]
        moving_variance = values["moving_variance"]

        rows.append({
            "Layer": layer_name,
            "Stage": stage,
            "Selected For Freezing": (
                layer_name in SELECTED_BN_LAYERS
            ),
            "Mean of moving mean": float(
                np.mean(moving_mean)
            ),
            "Mean absolute moving mean": float(
                np.mean(np.abs(moving_mean))
            ),
            "Average moving variance": float(
                np.mean(moving_variance)
            ),
            "Minimum moving variance": float(
                np.min(moving_variance)
            ),
            "Maximum moving variance": float(
                np.max(moving_variance)
            )
        })

    return rows


def compare_bn_statistics(before, after):

    rows = []

    for layer_name in before:

        mean_unchanged = np.array_equal(
            before[layer_name]["moving_mean"],
            after[layer_name]["moving_mean"]
        )

        variance_unchanged = np.array_equal(
            before[layer_name]["moving_variance"],
            after[layer_name]["moving_variance"]
        )

        rows.append({
            "Layer": layer_name,
            "Selected For Freezing": (
                layer_name in SELECTED_BN_LAYERS
            ),
            "Moving mean unchanged": mean_unchanged,
            "Moving variance unchanged": variance_unchanged,
            "Mean absolute difference": float(
                np.mean(
                    np.abs(
                        after[layer_name]["moving_mean"]
                        - before[layer_name]["moving_mean"]
                    )
                )
            ),
            "Variance absolute difference": float(
                np.mean(
                    np.abs(
                        after[layer_name]["moving_variance"]
                        - before[layer_name]["moving_variance"]
                    )
                )
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# Training History Plot
# ============================================================

def plot_history(history):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

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

    fig.suptitle(MODEL_NAME)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "training_history.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)

    print(f"Training history plot saved: {output_path}")


# ============================================================
# Load Training and Validation Data Only
# ============================================================

def load_training_data():

    baseline.section(
        "Load and Preprocess Training Data"
    )

    # Deliberately do not call baseline.load_data().
    # That function also loads the held-out test dataset.

    train_images, train_labels = baseline.preprocess_dataset(
        baseline.training_folder_path,
        size=IMAGE_SIZE
    )

    x_train_full = train_images.astype(np.float32)

    y_train_full = baseline.to_categorical(
        train_labels,
        num_classes=len(baseline.classes)
    )

    x_train, x_val, y_train, y_val = baseline.train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        stratify=train_labels,
        random_state=SEED
    )

    print("\nDataset split:")
    print(f"Training images:   {len(x_train)}")
    print(f"Validation images: {len(x_val)}")
    print(f"Image shape:       {x_train.shape[1:]}")
    print(f"Data type:         {x_train.dtype}")
    print(
        f"Pixel range:       "
        f"{x_train.min():.0f}–{x_train.max():.0f}"
    )

    print("\nInput preprocessing:")
    print("RGB images, float32, pixel range 0–255.")
    print("No division by 255; EfficientNetB0 has internal rescaling.")

    del train_images, x_train_full
    gc.collect()

    return x_train, y_train, x_val, y_val


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():

    keras.utils.set_random_seed(SEED)

    baseline.section(
        "EfficientNetB0 Scratch with Selectively Frozen BN"
    )

    print(f"Seed: {SEED}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Results directory: {RESULTS_DIR}")

    print("\nSelected frozen BN layers:")
    for layer_name in sorted(SELECTED_BN_LAYERS):
        print(f"  - {layer_name}")

    # --------------------------------------------------------
    # 1. Load training and validation data
    # --------------------------------------------------------

    x_train, y_train, x_val, y_val = load_training_data()

    # --------------------------------------------------------
    # 2. Build model
    # --------------------------------------------------------

    baseline.section(
        "Build EfficientNetB0 with Selective BN Freezing"
    )

    model, base = build_model()

    bn_layers = get_bn_layers(base)

    frozen_bn_layers = [
        layer
        for layer in bn_layers
        if not layer.trainable
    ]

    trainable_bn_layers = [
        layer
        for layer in bn_layers
        if layer.trainable
    ]

    trainable_conv_layers = [
        layer
        for layer in base.layers
        if isinstance(
            layer,
            (
                keras.layers.Conv2D,
                keras.layers.DepthwiseConv2D
            )
        )
        and layer.trainable
    ]

    print(f"\nTotal backbone layers: {len(base.layers)}")
    print(f"Total BN layers: {len(bn_layers)}")
    print(f"Frozen BN layers: {len(frozen_bn_layers)}")
    print(f"Trainable BN layers: {len(trainable_bn_layers)}")
    print(f"Trainable convolution layers: {len(trainable_conv_layers)}")

    print("\nFrozen BN layer names:")
    for layer in frozen_bn_layers:
        print(f"  - {layer.name}")

    # Verify exact layer configuration.
    actual_frozen_names = {
        layer.name
        for layer in frozen_bn_layers
    }

    if actual_frozen_names != SELECTED_BN_LAYERS:
        raise RuntimeError(
            "Frozen BN configuration does not match "
            "SELECTED_BN_LAYERS."
        )

    if len(frozen_bn_layers) != len(SELECTED_BN_LAYERS):
        raise RuntimeError(
            "Unexpected number of frozen BN layers."
        )

    if not trainable_bn_layers:
        raise RuntimeError(
            "No remaining BN layers are trainable."
        )

    if not trainable_conv_layers:
        raise RuntimeError(
            "No convolutional layers are trainable."
        )

    trainable_params = baseline.count_trainable(model)

    print(f"\nTrainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # 3. Capture initial BN statistics and a convolution kernel
    # --------------------------------------------------------

    bn_before = capture_bn_statistics(base)

    bn_before_rows = summarize_bn_statistics(
        bn_before,
        "Before training"
    )

    first_conv = trainable_conv_layers[0]

    conv_kernel_before = first_conv.kernel.numpy().copy()

    # --------------------------------------------------------
    # 4. Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        "efficientnetb0_scratch_selective_bn_best.keras"
    )

    callbacks = [
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

    # --------------------------------------------------------
    # 5. Train
    # --------------------------------------------------------

    baseline.section(
        "Train EfficientNetB0 Scratch with Selective BN Freezing"
    )

    model.summary()

    start_time = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=baseline.balanced_class_weights(y_train),
        callbacks=callbacks,
        verbose=2
    )

    training_time = (
        time.perf_counter() - start_time
    ) / 60

    best_epoch = int(
        np.argmin(history.history["val_loss"])
    ) + 1

    print(f"\nBest epoch: {best_epoch}")
    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Training time: {training_time:.2f} minutes")

    # --------------------------------------------------------
    # 6. Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(history.history)

    history_df.insert(
        0,
        "epoch",
        range(1, len(history_df) + 1)
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print(f"Training history saved: {history_path}")

    plot_history(history)

    # --------------------------------------------------------
    # 7. Verify BN statistics
    # --------------------------------------------------------

    bn_after = capture_bn_statistics(base)

    bn_comparison = compare_bn_statistics(
        bn_before,
        bn_after
    )

    bn_comparison_path = os.path.join(
        RESULTS_DIR,
        "batch_norm_verification.csv"
    )

    bn_comparison.to_csv(
        bn_comparison_path,
        index=False
    )

    frozen_rows = bn_comparison[
        bn_comparison["Selected For Freezing"]
    ]

    trainable_rows = bn_comparison[
        ~bn_comparison["Selected For Freezing"]
    ]

    frozen_bn_unchanged = (
        frozen_rows["Moving mean unchanged"].all()
        and frozen_rows["Moving variance unchanged"].all()
    )

    trainable_bn_changed_count = int(
        (
            ~trainable_rows["Moving mean unchanged"]
            | ~trainable_rows["Moving variance unchanged"]
        ).sum()
    )

    print("\nFrozen BN verification:")
    print(
        frozen_rows[
            [
                "Layer",
                "Moving mean unchanged",
                "Moving variance unchanged"
            ]
        ].to_string(index=False)
    )

    print(
        "\nAll selected BN statistics unchanged:",
        frozen_bn_unchanged
    )

    print("\nRemaining trainable BN layers:")
    print(f"Total: {len(trainable_rows)}")
    print(
        "Layers with changed moving statistics:",
        trainable_bn_changed_count
    )

    if not frozen_bn_unchanged:
        raise RuntimeError(
            "One or more selected frozen BN statistics changed."
        )

    if trainable_bn_changed_count == 0:
        raise RuntimeError(
            "None of the remaining trainable BN layers updated "
            "their moving statistics."
        )

    # Save before/after layer statistics.
    bn_after_rows = summarize_bn_statistics(
        bn_after,
        "After training"
    )

    bn_statistics_df = pd.DataFrame(
        bn_before_rows + bn_after_rows
    )

    bn_statistics_path = os.path.join(
        RESULTS_DIR,
        "batch_norm_layer_statistics.csv"
    )

    bn_statistics_df.to_csv(
        bn_statistics_path,
        index=False
    )

    # --------------------------------------------------------
    # 8. Verify convolutional weights were updated
    # --------------------------------------------------------

    conv_kernel_after = first_conv.kernel.numpy().copy()

    conv_weights_changed = not np.array_equal(
        conv_kernel_before,
        conv_kernel_after
    )

    print(f"\nConvolution layer checked: {first_conv.name}")
    print(f"Convolution weights changed: {conv_weights_changed}")

    if not conv_weights_changed:
        raise RuntimeError(
            "The checked convolution kernel did not change."
        )

    # --------------------------------------------------------
    # 9. Save experiment summary
    # --------------------------------------------------------

    summary = {
        "Model": MODEL_NAME,
        "Initialization": "Random (weights=None)",
        "BatchNormalization": "Selective freezing",
        "Frozen BN Layers": len(frozen_bn_layers),
        "Frozen BN Layer Names": ", ".join(
            sorted(SELECTED_BN_LAYERS)
        ),
        "Trainable BN Layers": len(trainable_bn_layers),
        "Trainable Convolution Layers": len(trainable_conv_layers),
        "Trainable Parameters": trainable_params,
        "Image Size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "Batch Size": BATCH_SIZE,
        "Learning Rate": LEARNING_RATE,
        "Maximum Epochs": MAX_EPOCHS,
        "Epochs Run": len(history.history["loss"]),
        "Best Epoch": best_epoch,
        "Best Validation Accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "Best Validation Loss": float(
            min(history.history["val_loss"])
        ),
        "Training Time (min)": training_time,
        "Selected BN Statistics Unchanged": bool(
            frozen_bn_unchanged
        ),
        "Trainable BN Layers With Changed Statistics": (
            trainable_bn_changed_count
        ),
        "Convolution Weights Changed": bool(
            conv_weights_changed
        )
    }

    summary_df = pd.DataFrame([summary])

    summary_path = os.path.join(
        RESULTS_DIR,
        "experiment_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nExperiment summary:")
    print(summary_df.to_string(index=False))

    print("\nNo test data was loaded or evaluated.")
    print(f"\nAll experiment outputs saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    run_experiment()
