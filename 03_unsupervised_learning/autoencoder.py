# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 2026

@author: Chitra Savaj
Unsupervised Learning: Convolutional Autoencoder (CAE)
Implements:
  1. Unsupervised representation learning via image reconstruction (MSE loss).
  2. Downstream transfer learning for brain tumor classification using the pre-trained encoder.
  3. Visual evaluation of MRI reconstructions and classification metrics (accuracy, F1, confusion matrix).
"""

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

# Project setup: locate root, register utils/preprocessing, set working directory
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _here if os.path.exists(os.path.join(_here, "model_utils.py")) else os.path.dirname(_here)
sys.path[:0] = [os.path.join(PROJECT_ROOT, d) for d in ("01_data_exploration_preprocessing", "utils")]
os.chdir(PROJECT_ROOT)

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from keras.models import Model
from sklearn.model_selection import train_test_split

from data_exploration import SEED, classes, section, show, testing_folder_path, training_folder_path
from model_utils import ModelTrainer, data_augmentation, free_gpu_memory, show_and_save
from preprocessing import image_size, normalize_and_encode, preprocess_dataset

import torch
torch.set_num_threads(4)

# ============================================================
# Configuration
# ============================================================
batch_size   = 32
ae_epochs    = 30
clf_epochs   = 30
patience     = 5
results_dir  = os.path.join(_here, "results", "autoencoder")

keras.utils.set_random_seed(SEED)


# ============================================================
# Object-Oriented Autoencoder Architecture
# ============================================================
class ConvAutoencoder:
    """Convolutional Autoencoder for unsupervised feature extraction and transfer learning.

    Encapsulates model construction, encoder feature extraction, downstream classifier
    adaptation, and reconstruction visualization.
    """

    def __init__(self, input_shape=(128, 128, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.classifier = None
        self._build_components()

    def _build_components(self):
        """Construct the encoder and decoder sub-networks and link them into an autoencoder."""
        # --- Encoder Network ---
        enc_inputs = Input(shape=self.input_shape, name="encoder_input")
        x = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(enc_inputs)
        x = MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)  # 64x64x32
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
        x = MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)  # 32x32x64
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
        enc_outputs = MaxPooling2D((2, 2), padding="same", name="enc_pool3")(x)  # 16x16x128 (Bottleneck)
        self.encoder = Model(enc_inputs, enc_outputs, name="encoder")

        # --- Decoder Network ---
        dec_inputs = Input(shape=enc_outputs.shape[1:], name="decoder_input")
        y = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same", name="dec_deconv1")(dec_inputs)
        y = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same", name="dec_deconv2")(y)
        y = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", name="dec_deconv3")(y)
        dec_outputs = Conv2D(self.input_shape[-1], (3, 3), activation="sigmoid", padding="same", name="dec_output")(y)
        self.decoder = Model(dec_inputs, dec_outputs, name="decoder")

        # --- End-to-End Autoencoder ---
        ae_outputs = self.decoder(self.encoder(enc_inputs))
        self.autoencoder = Model(enc_inputs, ae_outputs, name="Conv_Autoencoder")

    def build_classifier(self, freeze_encoder=True):
        """Build downstream classifier using the pre-trained encoder backbone (Transfer Learning)."""
        clf_inputs = Input(shape=self.input_shape, name="classifier_input")
        aug = data_augmentation()(clf_inputs)

        # Freeze or unfreeze encoder layers based on transfer learning strategy
        self.encoder.trainable = not freeze_encoder
        features = self.encoder(aug)

        x = GlobalAveragePooling2D(name="clf_gap")(features)
        x = Dense(128, activation="relu", name="clf_dense1")(x)
        x = Dropout(0.5, name="clf_dropout")(x)
        clf_outputs = Dense(self.num_classes, activation="softmax", name="clf_output")(x)

        mode_name = "Transfer_Frozen" if freeze_encoder else "Transfer_FineTuned"
        self.classifier = Model(clf_inputs, clf_outputs, name=f"Autoencoder_{mode_name}")
        return self.classifier

    def plot_reconstructions(self, x_test, num_images=8, save_path=None):
        """Plot a grid comparing original test MRI images with their reconstructed counterparts."""
        indices = np.linspace(0, len(x_test) - 1, num_images).astype(int)
        samples = x_test[indices]
        reconstructions = self.autoencoder.predict(samples, verbose=0)

        plt.figure(figsize=(16, 4))
        for i in range(num_images):
            # Original Image
            ax1 = plt.subplot(2, num_images, i + 1)
            plt.imshow(samples[i])
            ax1.set_title("Original", fontsize=9)
            plt.axis("off")

            # Reconstructed Image
            ax2 = plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(np.clip(reconstructions[i], 0, 1))
            ax2.set_title("Reconstructed", fontsize=9)
            plt.axis("off")

        plt.suptitle("Convolutional Autoencoder: Original vs. Reconstructed Test MRIs", fontsize=13)
        plt.tight_layout()
        show_and_save(save_path)


# ============================================================
# Main Execution Pipeline
# ============================================================
if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Step 1: Load and Preprocess the Dataset
    # ------------------------------------------------------------
    section("Step 1: Load and Preprocess the Dataset")
    train_images, train_labels = preprocess_dataset(training_folder_path, size=image_size)
    test_images,  test_labels  = preprocess_dataset(testing_folder_path, size=image_size)

    x_train_full, y_train_full = normalize_and_encode(train_images, train_labels)
    x_test,       y_test       = normalize_and_encode(test_images,  test_labels)

    # ------------------------------------------------------------
    # Step 2: Split Training Data into Train and Validation
    # ------------------------------------------------------------
    section("Step 2: Split Data for Unsupervised & Downstream Tasks")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.20, stratify=train_labels, random_state=SEED
    )

    show("Dataset splits", pd.DataFrame({
        "Images": [len(x_train), len(x_val), len(x_test)],
        "Image shape": [x_train.shape[1:], x_val.shape[1:], x_test.shape[1:]],
        "Labels shape": [y_train.shape[1:], y_val.shape[1:], y_test.shape[1:]]},
        index=["Training", "Validation", "Testing"]))

    # ------------------------------------------------------------
    # Step 3: Unsupervised Pre-training (Image Reconstruction)
    # ------------------------------------------------------------
    section("Step 3: Unsupervised Training - Convolutional Autoencoder")
    cae = ConvAutoencoder(input_shape=(*image_size, 3), num_classes=len(classes))
    ae_checkpoint = os.path.join(results_dir, "convolutional_autoencoder_best.keras")

    if os.path.exists(ae_checkpoint):
        print(f"Loading trained autoencoder from {ae_checkpoint}...")
        cae.autoencoder = keras.models.load_model(ae_checkpoint)
        cae.encoder = cae.autoencoder.get_layer("encoder")
    else:
        # In unsupervised learning: target is input itself (y = x), loss is MSE, no accuracy metric
        ae_trainer = ModelTrainer(
            model=cae.autoencoder,
            model_name="Convolutional Autoencoder",
            x_train=x_train, y_train=x_train,
            x_val=x_val,     y_val=x_val,
            x_test=x_test,   y_test=x_test,
            class_names=None,
            optimizer="adam",
            loss_function="mse",
            metrics=[],
            batch_size=batch_size,
            no_epochs=ae_epochs,
            patience=patience,
            results_dir=results_dir,
            verbosity=2,
        )
        ae_history = ae_trainer.train()
        ae_score, _ = ae_trainer.evaluate()

    cae.autoencoder.compile(optimizer="adam", loss="mse")
    ae_test_loss = cae.autoencoder.evaluate(x_test, x_test, verbose=0)
    print(f"\nAutoencoder Test Reconstruction MSE Loss: {ae_test_loss:.5f}")

    # Visual inspection of reconstructions on unseen test images
    cae.plot_reconstructions(
        x_test,
        num_images=8,
        save_path=os.path.join(results_dir, "autoencoder_reconstructions.png")
    )

    # ------------------------------------------------------------
    # Step 4: Transfer Learning - Downstream Classification
    # ------------------------------------------------------------
    section("Step 4: Transfer Learning - Supervised Classifier with Pre-trained Encoder")
    # Build classifier using pre-trained encoder weights (Feature Extractor: Frozen backbone)
    cae.build_classifier(freeze_encoder=True)

    clf_trainer = ModelTrainer(
        model=cae.classifier,
        model_name="Autoencoder Transfer Classifier",
        x_train=x_train, y_train=y_train,
        x_val=x_val,     y_val=y_val,
        x_test=x_test,   y_test=y_test,
        class_names=classes,
        class_weight="balanced",
        optimizer="adam",
        loss_function="categorical_crossentropy",
        metrics=["accuracy"],
        batch_size=batch_size,
        no_epochs=clf_epochs,
        patience=patience,
        results_dir=results_dir,
        verbosity=2,
    )

    clf_history = clf_trainer.train()
    clf_score, clf_predictions = clf_trainer.evaluate()

    # Free memory
    free_gpu_memory()
    print("\n[SUCCESS] Unsupervised Autoencoder & Transfer Learning pipeline completed successfully.")
