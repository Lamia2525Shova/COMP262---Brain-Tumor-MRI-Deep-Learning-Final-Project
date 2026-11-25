
# -*- coding: utf-8 -*-
"""
@author: Abdallah Divkar

DenseNet121 experiments for Brain Tumor MRI Classification.

Experiments:
    1. ImageNet pretrained - frozen backbone
    2. ImageNet pretrained - fine-tune final dense block (conv5_block)
    3. Random initialization - train from scratch

Uses the existing project preprocessing pipeline.
"""

import os
import gc
import time

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from keras import Input, Model
from keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import sys 

# Resolve the project root regardless of the working directory.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

os.chdir(PROJECT_ROOT)

# Make the existing project modules importable.
sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "01_data_exploration_preprocessing")
)

sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "utils")
)

from data_exploration import classes, SEED
from preprocessing import preprocess_dataset
from model_utils import data_augmentation, balanced_class_weights


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 32
PATIENCE = 5

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20
SCRATCH_EPOCHS = 30

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
SCRATCH_LR = 1e-3

DROPOUT_RATE = 0.4

RESULTS_DIR = os.path.join(
    "04_pretrained_model_finetuning",
    "results",
    "pretrained_densenet121_128"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

keras.utils.set_random_seed(SEED)


# ============================================================
# 2. DEVICE CHECK
# ============================================================

def check_gpu():

    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print("Keras version:", keras.__version__)
    print("Keras backend:", keras.backend.backend())
    print("PyTorch version:", torch.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Activate the centennial_ml environment and verify "
            "the CUDA-enabled PyTorch installation."
        )

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. MEMORY CLEANUP
# ============================================================

def free_gpu_memory():

    keras.backend.clear_session()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 4. BUILD DENSENET121
# ============================================================

def build_densenet121(
    weights="imagenet",
    name="densenet121_model",
    image_size=IMAGE_SIZE
):

    base = DenseNet121(
        weights=weights,
        include_top=False,
        input_shape=(*image_size, 3)
    )

    inputs = Input(shape=(*image_size, 3))

    x = data_augmentation()(inputs)

    if weights is not None:
        # Keep pretrained BatchNormalization statistics in inference mode.
        x = base(x, training=False)
    else:
        # Scratch model: train the backbone normally.
        x = base(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATE)(x)

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
# 5. FREEZE BACKBONE
# ============================================================

def freeze_base(model):

    base = model.get_layer("densenet121")
    base.trainable = False

    return model


# ============================================================
# 6. FINE-TUNE FINAL DENSENET BLOCK
# ============================================================

def unfreeze_last_block(model):

    base = model.get_layer("densenet121")

    # Allow selected layers in the backbone to become trainable.
    base.trainable = True

    trainable_layer_names = []

    for layer in base.layers:

        if isinstance(layer, keras.layers.BatchNormalization):

            # Keep all BatchNormalization layers frozen.
            layer.trainable = False

        elif layer.name.startswith("conv5_block"):

            # Unfreeze the final DenseNet dense block.
            layer.trainable = True
            trainable_layer_names.append(layer.name)

        else:

            # Keep earlier DenseNet layers frozen.
            layer.trainable = False

    if not trainable_layer_names:
        raise RuntimeError(
            "No conv5_block layers were found. "
            "Check the installed DenseNet121 architecture."
        )

    print("\nFine-tuning configuration")
    print("-" * 50)
    print("Unfrozen layers:", len(trainable_layer_names))
    print("First unfrozen layer:", trainable_layer_names[0])
    print("Last unfrozen layer:", trainable_layer_names[-1])

    trainable_params = sum(
        np.prod(variable.shape)
        for variable in model.trainable_weights
    )

    total_params = model.count_params()

    print("Trainable parameters:", f"{trainable_params:,}")
    print("Total parameters:", f"{total_params:,}")

    return model


# ============================================================
# 7. LOAD AND PREPROCESS DATA
# ============================================================

def load_data():

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_images, train_labels = preprocess_dataset(
        "./data/Training/",
        crop=True,
        size=IMAGE_SIZE,
        remove_duplicates=True
    )

    test_images, test_labels = preprocess_dataset(
        "./data/Testing/",
        crop=True,
        size=IMAGE_SIZE,
        remove_duplicates=True
    )

    print("Original training array:", train_images.shape)
    print("Original testing array:", test_images.shape)

    # DenseNet121 uses its own ImageNet preprocessing.
    # Do not divide these images by 255.
    x_train_full = preprocess_input(
        train_images.astype(np.float32)
    )

    x_test = preprocess_input(
        test_images.astype(np.float32)
    )

    y_train_full = keras.utils.to_categorical(
        train_labels,
        num_classes=len(classes)
    )

    y_test = keras.utils.to_categorical(
        test_labels,
        num_classes=len(classes)
    )

    # Use the same stratified 80/20 split as EfficientNetB0.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=train_labels
    )

    print("\nDataset split")
    print("-" * 50)
    print("Training:", x_train.shape)
    print("Validation:", x_val.shape)
    print("Testing:", x_test.shape)

    print(
        "Preprocessed training pixel range:",
        f"{x_train.min():.2f} to {x_train.max():.2f}"
    )

    del train_images, test_images, x_train_full
    gc.collect()

    return x_train, x_val, y_train, y_val, x_test, y_test


# ============================================================
# 8. TRAIN MODEL
# ============================================================

def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    model_name,
    epochs,
    learning_rate,
    class_weights
):

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_best.keras"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    trainable_params = sum(
        np.prod(variable.shape)
        for variable in model.trainable_weights
    )

    print("Trainable parameters:", f"{trainable_params:,}")
    print("Learning rate:", learning_rate)
    print("Maximum epochs:", epochs)

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weights,
        verbose=1
    )

    training_time = (time.time() - start_time) / 60

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    best_val_loss = float(
        min(history.history["val_loss"])
    )

    best_val_accuracy = float(
        history.history["val_accuracy"][
            best_epoch - 1
        ]
    )

    print("\nTraining completed")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Best validation accuracy:", best_val_accuracy)
    print("Training time:", f"{training_time:.2f} minutes")

    # Load the best saved checkpoint.
    best_model = keras.models.load_model(checkpoint_path)

    del model
    free_gpu_memory()

    return (
        best_model,
        history,
        checkpoint_path,
        training_time,
        best_epoch,
        trainable_params,
        best_val_accuracy,
        best_val_loss
    )


# ============================================================
# 9. PLOT TRAINING HISTORY
# ============================================================

def plot_history(history, model_name):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history.history["loss"],
        label="Training Loss"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_training_history.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# 10. EVALUATE ON HELD-OUT TEST SET
# ============================================================

def evaluate_test(
    model,
    x_test,
    y_test,
    model_name
):

    print("\n" + "=" * 60)
    print("TEST EVALUATION:", model_name)
    print("=" * 60)

    probabilities = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report")
    print(report)

    report_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(classes))
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Save predictions for further analysis.
    predictions_df = pd.DataFrame({
        "true_label": [classes[i] for i in y_true],
        "predicted_label": [classes[i] for i in y_pred],
        "confidence": np.max(probabilities, axis=1)
    })

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision_macro": precision,
        "Recall_macro": recall,
        "F1_macro": macro_f1
    }


# ============================================================
# 11. RUN ALL EXPERIMENTS
# ============================================================

def run_experiment():

    check_gpu()

    (
        x_train,
        x_val,
        y_train,
        y_val,
        x_test,
        y_test
    ) = load_data()

    class_weights = balanced_class_weights(y_train)

    experiments = []

    # --------------------------------------------------------
    # STAGE 1: IMAGENET PRETRAINED - FROZEN BACKBONE
    # --------------------------------------------------------

    print("\n\nSTAGE 1: FROZEN PRETRAINED DENSENET121")

    stage1_model = build_densenet121(
        weights="imagenet",
        name="densenet121_stage1"
    )

    freeze_base(stage1_model)

    (
        stage1_model,
        stage1_history,
        stage1_checkpoint,
        stage1_time,
        stage1_best_epoch,
        stage1_params,
        stage1_val_accuracy,
        stage1_val_loss
    ) = train_model(
        model=stage1_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_name="densenet121_stage1_frozen",
        epochs=STAGE1_EPOCHS,
        learning_rate=STAGE1_LR,
        class_weights=class_weights
    )

    plot_history(
        stage1_history,
        "densenet121_stage1_frozen"
    )

    experiments.append({
        "name": "densenet121_stage1_frozen",
        "checkpoint": stage1_checkpoint,
        "history": stage1_history,
        "training_time": stage1_time,
        "best_epoch": stage1_best_epoch,
        "trainable_params": stage1_params,
        "val_accuracy": stage1_val_accuracy,
        "val_loss": stage1_val_loss
    })

    # --------------------------------------------------------
    # STAGE 2: FINE-TUNE FINAL DENSE BLOCK
    # --------------------------------------------------------

    print("\n\nSTAGE 2: FINE-TUNING DENSENET121")

    # Continue from the best Stage 1 checkpoint.
    stage2_model = stage1_model

    unfreeze_last_block(stage2_model)

    (
        stage2_model,
        stage2_history,
        stage2_checkpoint,
        stage2_time,
        stage2_best_epoch,
        stage2_params,
        stage2_val_accuracy,
        stage2_val_loss
    ) = train_model(
        model=stage2_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_name="densenet121_stage2_finetuned",
        epochs=STAGE2_EPOCHS,
        learning_rate=STAGE2_LR,
        class_weights=class_weights
    )

    plot_history(
        stage2_history,
        "densenet121_stage2_finetuned"
    )

    experiments.append({
        "name": "densenet121_stage2_finetuned",
        "checkpoint": stage2_checkpoint,
        "history": stage2_history,
        "training_time": stage2_time,
        "best_epoch": stage2_best_epoch,
        "trainable_params": stage2_params,
        "val_accuracy": stage2_val_accuracy,
        "val_loss": stage2_val_loss
    })

    del stage1_model, stage2_model
    free_gpu_memory()

    # --------------------------------------------------------
    # STAGE 3: TRAIN FROM SCRATCH
    # --------------------------------------------------------

    print("\n\nSTAGE 3: DENSENET121 FROM SCRATCH")

    scratch_model = build_densenet121(
        weights=None,
        name="densenet121_from_scratch"
    )

    (
        scratch_model,
        scratch_history,
        scratch_checkpoint,
        scratch_time,
        scratch_best_epoch,
        scratch_params,
        scratch_val_accuracy,
        scratch_val_loss
    ) = train_model(
        model=scratch_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_name="densenet121_from_scratch",
        epochs=SCRATCH_EPOCHS,
        learning_rate=SCRATCH_LR,
        class_weights=class_weights
    )

    plot_history(
        scratch_history,
        "densenet121_from_scratch"
    )

    experiments.append({
        "name": "densenet121_from_scratch",
        "checkpoint": scratch_checkpoint,
        "history": scratch_history,
        "training_time": scratch_time,
        "best_epoch": scratch_best_epoch,
        "trainable_params": scratch_params,
        "val_accuracy": scratch_val_accuracy,
        "val_loss": scratch_val_loss
    })

    del scratch_model
    free_gpu_memory()

    # --------------------------------------------------------
    # TEST EVALUATION AFTER ALL TRAINING IS COMPLETE
    # --------------------------------------------------------

    print("\n\nEVALUATING ALL EXPERIMENTS ON TEST SET")

    comparison_results = []

    for experiment in experiments:

        print("\nLoading:", experiment["name"])

        model = keras.models.load_model(
            experiment["checkpoint"]
        )

        metrics = evaluate_test(
            model=model,
            x_test=x_test,
            y_test=y_test,
            model_name=experiment["name"]
        )

        metrics["Trainable Parameters"] = experiment["trainable_params"]
        metrics["Best Epoch"] = experiment["best_epoch"]
        metrics["Training Time (min)"] = experiment["training_time"]
        metrics["Best Validation Accuracy"] = experiment["val_accuracy"]
        metrics["Best Validation Loss"] = experiment["val_loss"]

        comparison_results.append(metrics)

        del model
        free_gpu_memory()

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(comparison_results)

    comparison_path = os.path.join(
        RESULTS_DIR,
        "densenet121_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("DENSENET121 EXPERIMENT COMPARISON")
    print("=" * 80)

    print(
        comparison_df.to_string(index=False)
    )

    print("\nResults saved to:")
    print(os.path.abspath(RESULTS_DIR))

    del x_train, x_val, y_train, y_val, x_test, y_test
    free_gpu_memory()


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    run_experiment()