# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:06:09 2026

@author: Haripriya Gounalan
This script builds, trains and evaluates a simple CNN
"""

import os
import sys
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Project setup: find the project root, make the shared folders importable and run from the root,
# so "./data" works from any folder (also on Kaggle, where all files are in one folder); results are saved next to each script
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _here if os.path.exists(os.path.join(_here, "model_utils.py")) else os.path.dirname(_here)
sys.path[:0] = [os.path.join(PROJECT_ROOT, d) for d in ("01_data_exploration_preprocessing", "utils")]
os.chdir(PROJECT_ROOT)

import keras
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from data_exploration import SEED, classes, section, show, testing_folder_path, training_folder_path
from model_utils import ModelTrainer, data_augmentation
from preprocessing import image_size, normalize_and_encode, preprocess_dataset

# ============================================================
# Configuration
# ============================================================
batch_size = 32
no_epochs  = 30
patience   = 5

keras.utils.set_random_seed(SEED)

if __name__ == "__main__":
    # ============================================================
    # Step 1: Load and Preprocess the Dataset
    # ============================================================
    section("Step 1: Load and Preprocess the Dataset")

    train_images, train_labels = preprocess_dataset(training_folder_path)          # crop and resize images
    test_images,  test_labels  = preprocess_dataset(testing_folder_path)

    x_train_full, y_train_full = normalize_and_encode(train_images, train_labels)   # normalize images and encode labels
    x_test,       y_test       = normalize_and_encode(test_images,  test_labels)

    # ============================================================
    # Step 2: Split Training Data into Training and Validation
    # ============================================================
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.20, stratify=train_labels, random_state=SEED)

    section("Step 2: Split Training Data into Training and Validation")
    show("Dataset splits", pd.DataFrame({
        "Images": [len(x_train), len(x_val), len(x_test)],
        "Image shape": [x_train.shape[1:], x_val.shape[1:], x_test.shape[1:]],
        "Labels shape": [y_train.shape[1:], y_val.shape[1:], y_test.shape[1:]]},
        index=["Training", "Validation", "Testing"]))

    # ============================================================
    # Step 3: Build, Train and Evaluate a Simple CNN - Base Model
    # ============================================================
    input_shape = (*image_size, 3)
    num_classes = len(classes)

    cnn_model = Sequential(name="Simple_CNN")
    cnn_model.add(Input(shape=input_shape))
    cnn_model.add(data_augmentation())                                  # rotate, shift and flip training images
    cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(num_classes, activation="softmax"))

    cnn_trainer = ModelTrainer(cnn_model, "Simple CNN",
                               x_train, y_train,
                               x_val,   y_val,
                               x_test,  y_test,
                               class_names=classes, class_weight="balanced",
                               batch_size=batch_size, no_epochs=no_epochs, patience=patience,
                               results_dir=os.path.join(_here, "results", "simple_cnn"),
                               verbosity=2)

    cnn_history = cnn_trainer.train()
    cnn_score, cnn_predictions = cnn_trainer.evaluate()
