# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 2026

@author: Haripriya Gounalan
This script tunes the fine-tuning stage of ResNet50 (the best model of pretrained_resnet50_128.py) with
Keras Tuner random search:
  Step 3 - Stage 1 is trained once (ImageNet weights frozen, only the new head is trained)
  Step 4 - every trial starts from that Stage 1 model and fine-tunes it with its own settings:
           learning rate, how many ResNet50 blocks are unfrozen, dropout rate, optimizer, batch size
  Step 5 - the best trial (lowest validation loss) is evaluated on the test data
Runs at 128x128; pretrained_resnet50_tuning_224.py runs the same code at 224x224.
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
sys.path[:0] = [os.path.join(PROJECT_ROOT, d) for d in ("01_data_exploration_preprocessing", "utils")] + [_here]
os.chdir(PROJECT_ROOT)

import keras
import keras_tuner as kt
import pandas as pd
from keras.layers import Dropout

from data_exploration import SEED, classes, section, show
from model_utils import ModelTrainer, balanced_class_weights, free_gpu_memory
from pretrained_resnet50_128 import (build_resnet50, count_trainable, freeze_base, load_data, patience,
                                     stage1_epochs, stage1_learning_rate, stage2_epochs, train_and_evaluate,
                                     unfreeze_blocks)

# ============================================================
# Configuration
# ============================================================
batch_size  = 32                    # Stage 1
batch_sizes = [16, 32]              # fine-tuning: searched (smaller batches = noisier updates, often better generalization)
max_trials  = 20                    # 144 combinations below: random search samples 20 of them

# fine-tuning settings searched by Keras Tuner
learning_rates  = [1e-5, 5e-5, 1e-4]                        # small steps: only adjust the pretrained weights
unfreeze_blocks_options = {"conv5": ("conv5_",),            # last block only (as in pretrained_resnet50_128.py)
                           "conv4+conv5": ("conv4_", "conv5_")}   # last two blocks
dropout_rates   = [0.2, 0.3, 0.4, 0.5]
optimizers      = {"adam":    lambda lr: keras.optimizers.Adam(learning_rate=lr),
                   "adamw":   lambda lr: keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),   # + L2-style weight decay
                   "rmsprop": lambda lr: keras.optimizers.RMSprop(learning_rate=lr)}

keras.utils.set_random_seed(SEED)

# ============================================================
# Random Search Tuner that Frees GPU Memory Before Each Trial
# ============================================================
class MemorySafeRandomSearch(kt.RandomSearch):
    """Random search that frees GPU memory before each trial, searches the batch size and saves each trial's epoch history."""

    def __init__(self, *args, batch_sizes=batch_sizes, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sizes = batch_sizes

    def run_trial(self, trial, *args, **kwargs):
        free_gpu_memory()
        kwargs["batch_size"] = trial.hyperparameters.Choice("batch_size", self.batch_sizes)   # batch size is a fit() setting
        trial_dir = self.get_trial_dir(trial.trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [                  # per-epoch history for the plots
            keras.callbacks.CSVLogger(os.path.join(trial_dir, "history.csv"))]
        return super().run_trial(trial, *args, **kwargs)


# ============================================================
# Build One Fine-Tuning Trial
# ============================================================
def fine_tuning_builder(stage1_path):
    """Return the Keras Tuner build function: each trial starts from the saved Stage 1 model."""

    def build_fine_tuning(hp):
        learning_rate = hp.Choice("learning_rate", learning_rates)
        unfreeze      = hp.Choice("unfreeze", list(unfreeze_blocks_options))
        dropout_rate  = hp.Choice("dropout", dropout_rates)
        optimizer     = hp.Choice("optimizer", list(optimizers))

        model = keras.models.load_model(stage1_path)                       # trained head, frozen ResNet50
        next(layer for layer in model.layers if isinstance(layer, Dropout)).rate = dropout_rate   # no weights: safe to change
        unfreeze_blocks(model, unfreeze_blocks_options[unfreeze])

        model.compile(optimizer=optimizers[optimizer](learning_rate),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    return build_fine_tuning


# ============================================================
# Save Tuning Results
# ============================================================
def export_tuning_results(tuner, results_dir):
    """Save and display every trial, best first."""
    rows = [{"Trial ID": trial.trial_id,
             "Val_Loss": trial.score,
             "Val_Accuracy": (trial.metrics.get_best_value("val_accuracy")
                              if trial.metrics.exists("val_accuracy") else None),
             **trial.hyperparameters.values}
            for trial in tuner.oracle.trials.values() if trial.status == "COMPLETED"]

    results_df = pd.DataFrame(rows).sort_values("Val_Loss")
    results_df.to_csv(os.path.join(results_dir, "keras_tuner_results.csv"), index=False)
    show("ResNet50 fine-tuning trials (best first)", results_df.set_index("Trial ID"))
    return results_df


def trial_history(tuner, trial):
    """Per-epoch history of a trial, for the training plots."""
    history = pd.read_csv(os.path.join(tuner.get_trial_dir(trial.trial_id), "history.csv")).drop(columns="epoch")
    return SimpleNamespace(history=history.to_dict("list"))


# ============================================================
# Run the Tuning at One Image Size
# ============================================================
def run_tuning(image_size=(128, 128), batch_size=batch_size, batch_sizes=batch_sizes, results_dir=None):
    """Steps 1-5 at the given image size; results go to results/pretrained_resnet50_tuning_<size>/.
    batch_size: Stage 1; batch_sizes: searched for fine-tuning (smaller for large images, to fit in GPU memory)."""
    results_dir = results_dir or os.path.join(_here, "results", f"pretrained_resnet50_tuning_{image_size[0]}")
    os.makedirs(results_dir, exist_ok=True)

    # Steps 1-2: same images, scaling and train/validation split as pretrained_resnet50_128.py
    data, test_images = load_data(image_size)
    x_train, y_train, x_val, y_val, x_test, y_test = data

    # ============================================================
    # Step 3: Stage 1 - Feature Extraction (trained once, shared by every trial)
    # ============================================================
    section("Step 3: Stage 1 - Feature Extraction (ResNet50 frozen)")

    stage1_path = os.path.join(results_dir, "resnet50_stage1_frozen_best.keras")
    if os.path.exists(stage1_path):
        print(f"Stage 1 model found, reused: {stage1_path}")              # a rerun skips Stage 1
    else:
        stage1_model = build_resnet50("imagenet", "ResNet50_Stage1_Frozen", image_size)
        freeze_base(stage1_model)
        print(f"Trainable parameters: {count_trainable(stage1_model):,} (classification head only)")
        stage1_trainer, _ = train_and_evaluate(
            stage1_model, "ResNet50 Stage1 Frozen", stage1_learning_rate, stage1_epochs, data, test_images, results_dir, batch_size)
        stage1_trainer.model = None                                       # saved to stage1_path by the checkpoint
        del stage1_model
        free_gpu_memory()

    # ============================================================
    # Step 4: Tune the Fine-Tuning Stage
    # ============================================================
    section("Step 4: Tune the Fine-Tuning Stage (random search)")

    tuner = MemorySafeRandomSearch(
        fine_tuning_builder(stage1_path), objective="val_loss", max_trials=max_trials, seed=SEED,
        batch_sizes=batch_sizes, directory=results_dir, project_name=f"random_search_resnet50_{image_size[0]}",
        overwrite=False)

    tuner.search_space_summary()

    # same early stopping and class weights as the other models; the learning rate is halved when val loss stalls
    tuner.search(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=stage2_epochs, class_weight=balanced_class_weights(y_train),       # batch size: chosen per trial
        callbacks=[keras.callbacks.EarlyStopping(
                       monitor="val_loss", patience=patience, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(                             # shorter patience than early stopping:
                       monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)],   # finer steps before giving up
        verbose=2)

    export_tuning_results(tuner, results_dir)

    # ============================================================
    # Step 5: Evaluate the Best Trial on Test Data (no retraining)
    # ============================================================
    section("Step 5: Evaluate the Best Fine-Tuned ResNet50")

    free_gpu_memory()
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_model = tuner.load_model(best_trial)                             # rebuilt from its settings + saved weights
    best_model.save(os.path.join(results_dir, "resnet50_tuned_best.keras"))
    with open(os.path.join(results_dir, "best_hyperparameters.json"), "w", encoding="utf-8") as file:
        json.dump(best_trial.hyperparameters.values, file, indent=2)

    show("Best fine-tuning settings", pd.Series(best_trial.hyperparameters.values, name="Value").to_frame())
    print(f"Trainable parameters: {count_trainable(best_model):,}")

    tuned_trainer = ModelTrainer(best_model, "ResNet50 Tuned",
                                 x_train, y_train,
                                 x_val,   y_val,
                                 x_test,  y_test,
                                 class_names=classes,
                                 display_images=test_images,             # original images, not the ResNet-scaled ones
                                 results_dir=results_dir)
    tuned_trainer.history = trial_history(tuner, best_trial)             # the search already trained it: plot its epochs
    tuned_trainer.evaluate()


if __name__ == "__main__":
    run_tuning((128, 128))
