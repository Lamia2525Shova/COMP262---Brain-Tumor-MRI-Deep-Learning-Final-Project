# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 2026

@author: Haripriya Gounalan
This script tunes a CNN with Keras Tuner Hyperband and evaluates the best model found by the search.
"""

import json
import os
import sys
from types import SimpleNamespace
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
import keras_tuner as kt
import pandas as pd
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from data_exploration import SEED, classes, section, show, testing_folder_path, training_folder_path
from model_utils import ModelTrainer, balanced_class_weights, data_augmentation, free_gpu_memory
from preprocessing import image_size, normalize_and_encode, preprocess_dataset

# ============================================================
# Configuration
# ============================================================
batch_size = 32
patience   = 5

max_epochs       = 30
hyperband_factor = 3

results_dir = os.path.join(_here, "results", "cnn_tuning")
project_name = "hyperband_narrowed"          # 2nd search, ranges narrowed from the 1st (hyperband_val_loss_gap_v2)

keras.utils.set_random_seed(SEED)

# ============================================================
# Hyperband Tuner that Frees GPU Memory Before Each Trial
# ============================================================
class MemorySafeHyperband(kt.Hyperband):
    """Hyperband that frees GPU memory before each trial and saves each trial's epoch history."""

    def run_trial(self, trial, *args, **kwargs):
        free_gpu_memory()
        trial_dir = self.get_trial_dir(trial.trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [                  # per-epoch history for the plots
            keras.callbacks.CSVLogger(os.path.join(trial_dir, "history.csv"))]
        return super().run_trial(trial, *args, **kwargs)


# ============================================================
# Build CNN
# ============================================================
def build_cnn_architecture(hp):
    """Build the CNN using the selected settings."""
    architecture = hp.Choice("architecture", ["standard", "double_conv", "batch_norm"])

    model = Sequential(name="Tuned_CNN")
    model.add(Input(shape=(*image_size, 3)))
    model.add(data_augmentation())

    # number of conv blocks
    conv_blocks = hp.Int("conv_blocks", 4, 6)                           # 1st search: only 5-6 blocks reached val loss < 0.5

    for i in range(conv_blocks):
        # block i settings exist only when conv_blocks > i (keeps unused blocks out of the trials table)
        with hp.conditional_scope("conv_blocks", list(range(i + 1, 7))):
            filters      = hp.Choice(f"filters_{i}", [32, 64] if i == 0 else [64, 128, 256])
            kernel_size  = hp.Choice(f"kernel_{i}", [3, 5])
            dropout_rate = hp.Float(f"conv_dropout_{i}", min_value=0.0, max_value=0.3, step=0.1)

        model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), activation="relu", padding="same"))

        # add a second convolution
        if architecture == "double_conv":
            model.add(Conv2D(filters, kernel_size=(3, 3), activation="relu", padding="same"))

        # add batch normalization
        if architecture == "batch_norm":
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # reduce the number of inputs to the dense layers
    model.add(GlobalAveragePooling2D())

    # one dense layer (1st search: the top 11 trials all used one)
    dense_units   = hp.Choice("dense_units_0", [128, 256, 512])
    dense_dropout = hp.Float("dense_dropout_0", min_value=0.2, max_value=0.4, step=0.1)

    model.add(Dense(dense_units, activation="relu"))
    model.add(Dropout(dense_dropout))

    # output layer
    model.add(Dense(len(classes), activation="softmax"))

    return model


# ============================================================
# Compile CNN
# ============================================================
def build_tuner_cnn(hp):
    """Choose the optimizer and compile the CNN."""
    model = build_cnn_architecture(hp)

    # optimizer tuning
    optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop"])        # SGD dropped: too slow for short Hyperband rounds
    learning_rate  = hp.Choice("learning_rate", [5e-4, 1e-3])            # 1e-4 dropped: too slow within 20 epochs

    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# ============================================================
# Show Best Hyperparameters
# ============================================================
def show_best_hyperparameters(hyperparameters):
    """Show the best settings."""
    rows = [{"Hyperparameter": k, "Value": v}
            for k, v in hyperparameters.values.items() if not k.startswith("tuner/")]

    show("Selected hyperparameters", pd.DataFrame(rows).set_index("Hyperparameter"))


# ============================================================
# Save Tuning Results
# ============================================================
def export_tuning_results(tuner):
    """Save and display the tuning results."""
    rows = []

    for trial in tuner.oracle.trials.values():
        row = {
            "Trial ID": trial.trial_id,
            "Status": trial.status,
            "Val_Loss": trial.score,
            "Val_Accuracy": (trial.metrics.get_best_value("val_accuracy")
                             if trial.metrics.exists("val_accuracy") else None),
            "Initial_Epoch": trial.hyperparameters.values.get("tuner/initial_epoch", 0),
            "Epoch_Budget": trial.hyperparameters.values.get("tuner/epochs", max_epochs),
            **{k: v for k, v in trial.hyperparameters.values.items() if not k.startswith("tuner/")}
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No tuning trials are available.")

    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(["Val_Loss", "Epoch_Budget"], ascending=[True, False], na_position="last")
    results_df.to_csv(os.path.join(results_dir, "keras_tuner_results.csv"), index=False)

    # display top trials
    summary_columns = ["Trial ID", "Status", "architecture", "Val_Loss", "Val_Accuracy", "Epoch_Budget"]
    show("Top 10 Hyperband trials", results_df.head(10)[summary_columns].set_index("Trial ID"))

    # choose one candidate from each architecture
    completed_df = results_df[results_df["Status"] == "COMPLETED"]
    best_architectures = (completed_df.dropna(subset=["Val_Loss"])
                         .drop_duplicates(subset=["architecture"]))
    best_architectures.to_csv(os.path.join(results_dir, "best_architectures.csv"), index=False)

    if best_architectures.empty:
        raise RuntimeError("No completed tuning trials have a validation loss.")

    show("Architecture candidates from Hyperband", best_architectures[
        ["architecture", "Trial ID", "Val_Loss", "Val_Accuracy", "Epoch_Budget"]].set_index("architecture"))

    return results_df, best_architectures


# ============================================================
# Compare Architectures (best saved Hyperband model of each, no retraining)
# ============================================================
def compare_architectures(tuner, candidates, x_val, y_val):
    """Load the best Hyperband model of each architecture and compare them on the validation data."""
    rows = []
    best_loss, best_trial = float("inf"), None

    for _, candidate in candidates.iterrows():
        trial = tuner.oracle.get_trial(candidate["Trial ID"])
        free_gpu_memory()
        model = tuner.load_model(trial)                     # rebuilt from the trial's settings + its saved weights
        score = model.evaluate(x_val, y_val, verbose=0, return_dict=True)
        rows.append({"Architecture": candidate["architecture"], "Trial ID": trial.trial_id,
                     "Val_Loss": score["loss"], "Val_Accuracy": score["accuracy"],
                     "Parameters": model.count_params()})
        if score["loss"] < best_loss:
            best_loss, best_trial = score["loss"], trial
        del model

    comparison_df = pd.DataFrame(rows).sort_values("Val_Loss")
    comparison_df.to_csv(os.path.join(results_dir, "architecture_comparison.csv"), index=False)
    show("Best model of each architecture (validation data)", comparison_df.set_index("Architecture"))

    free_gpu_memory()
    best_model = tuner.load_model(best_trial)
    best_model.save(os.path.join(results_dir, "tuned_cnn_best.keras"))
    with open(os.path.join(results_dir, "best_hyperparameters.json"), "w", encoding="utf-8") as file:
        json.dump({k: v for k, v in best_trial.hyperparameters.values.items() if not k.startswith("tuner/")}, file, indent=2)
    return best_model, best_trial


def trial_history(tuner, trial):
    """Per-epoch history of a trial for the training plots (joined with the trials it continued from)."""
    frames = []
    while trial is not None:
        frames.insert(0, pd.read_csv(os.path.join(tuner.get_trial_dir(trial.trial_id), "history.csv")))
        parent = trial.hyperparameters.values.get("tuner/trial_id")       # Hyperband continued this trial from another
        trial = tuner.oracle.get_trial(parent) if parent else None
    history = pd.concat(frames, ignore_index=True).drop(columns="epoch")
    return SimpleNamespace(history=history.to_dict("list"))


if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)

    # ============================================================
    # Step 1: Load and Preprocess Data
    # ============================================================
    section("Step 1: Load and Preprocess Data")

    train_images, train_labels = preprocess_dataset(training_folder_path)
    test_images,  test_labels  = preprocess_dataset(testing_folder_path)

    x_train_full, y_train_full = normalize_and_encode(train_images, train_labels)
    x_test,       y_test       = normalize_and_encode(test_images,  test_labels)

    # ============================================================
    # Step 2: Split Train and Validation Data
    # ============================================================
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.20, stratify=train_labels, random_state=SEED)

    section("Step 2: Split Train and Validation Data")

    show("Dataset splits", pd.DataFrame({
        "Images": [len(x_train), len(x_val), len(x_test)],
        "Image shape": [x_train.shape[1:], x_val.shape[1:], x_test.shape[1:]],
        "Labels shape": [y_train.shape[1:], y_val.shape[1:], y_test.shape[1:]]},
        index=["Training", "Validation", "Testing"]))

    # ============================================================
    # Step 3: Tune CNN
    # ============================================================
    section("Step 3: Tune CNN")

    tuner = MemorySafeHyperband(
        build_tuner_cnn, objective="val_loss",
        max_epochs=max_epochs, factor=hyperband_factor, seed=SEED,
        directory=results_dir, project_name=project_name, overwrite=False)

    tuner.search_space_summary()

    # tuner search with early stopping
    tuner.search(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=max_epochs, batch_size=batch_size, class_weight=balanced_class_weights(y_train),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)],
        verbose=2)

    tuner.results_summary(num_trials=10)
    results_df, best_architectures = export_tuning_results(tuner)

    # ============================================================
    # Step 4: Compare Architectures and Evaluate the Best CNN on Test Data
    # ============================================================
    section("Step 4: Compare Architectures and Evaluate the Best CNN")

    free_gpu_memory()                                       # release the last tuning trial
    best_model, best_trial = compare_architectures(tuner, best_architectures, x_val, y_val)

    section("Best CNN Hyperparameters")
    show_best_hyperparameters(best_trial.hyperparameters)

    tuned_trainer = ModelTrainer(best_model, "Tuned CNN",
                                 x_train, y_train,
                                 x_val,   y_val,
                                 x_test,  y_test,
                                 class_names=classes, results_dir=results_dir)
    tuned_trainer.history = trial_history(tuner, best_trial)      # the search already trained it: plot its epochs
    tuned_score, tuned_predictions = tuned_trainer.evaluate()
