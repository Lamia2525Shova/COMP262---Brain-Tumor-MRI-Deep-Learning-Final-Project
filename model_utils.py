"""
Created on Thu Nov  5 21:31:24 2026

@author: Haripriya Gounalan
Common helpers for every model in the project: supervised, unsupervised and state-of-the-art (pretrained)."""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")         # Keras on PyTorch (TensorFlow has no GPU support on native Windows)

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Default Training Configuration
# ============================================================
batch_size = 32
no_epochs  = 30
patience   = 5  # Early stopping


# ============================================================
# Data Augmentation (Keras 3 replacement for ImageDataGenerator)
# ============================================================
def data_augmentation():
    """Random rotation, shift and flip on each training batch (inactive during evaluation and prediction)."""
    return keras.Sequential([
        keras.layers.RandomRotation(10 / 360, fill_mode="nearest"),                # +/- 10 degrees (factor = fraction of 360)
        keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),           # +/- 5% height and width shift
        keras.layers.RandomFlip("horizontal"),
    ], name="data_augmentation")


# ============================================================
# Helper Functions for Visualizing
# ============================================================
def plot_training_history(history, metric, plot_title):
    """Plot a training vs validation metrics."""
    train, val = history.history[metric], history.history[f"val_{metric}"]
    epochs     = range(1, len(train) + 1)
    fmt        = (lambda v: f"{v:.0%}") if metric == "accuracy" else (lambda v: f"{v:.3f}")
    name       = metric.capitalize()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train, "o-", label=f"Training {name} ({fmt(train[-1])})")
    plt.plot(epochs, val,   "o-", label=f"Validation {name} ({fmt(val[-1])})")

    for epoch, t, v in zip(epochs, train, val):
        plt.text(epoch, t, fmt(t), ha="center", va="bottom", fontsize=7)
        plt.text(epoch, v, fmt(v), ha="center", va="top",    fontsize=7)

    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(plot_title)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_image(df, class_names, total_images=8, subplot_size=(2, 8), prediction_distribution=None, figsize=(24, 6)):
    """Display each image with its true class, followed by its prediction distribution if provided."""
    num_classes = len(class_names)
    plt.figure(figsize=figsize)

    for i in range(total_images):
        label      = df["labels"][i]
        true_label = int(label) if np.ndim(label) == 0 else int(np.argmax(label))     # Integer or one-hot labels

        # Set image position
        position = 2 * i + 1 if prediction_distribution is not None else i + 1

        # Plot image
        plt.subplot(subplot_size[0], subplot_size[1], position)
        plt.imshow(df["images"][i], cmap="gray")
        plt.xlabel(f"{class_names[true_label]} ({true_label})", fontsize=9)
        plt.xticks([])
        plt.yticks([])

        # Plot prediction probability distribution
        if prediction_distribution is not None:
            probabilities   = prediction_distribution[i]
            predicted_label = int(np.argmax(probabilities))
            ax   = plt.subplot(subplot_size[0], subplot_size[1], position + 1)
            bars = plt.bar(range(num_classes), probabilities, color="gray")
            bars[true_label].set_color("green")
            if predicted_label != true_label:
                bars[predicted_label].set_color("blue")

            bars[predicted_label].set_edgecolor("blue")                                 # Always outline the prediction
            bars[predicted_label].set_linewidth(2.5)
            ax.set_title(f"Predicted: {class_names[predicted_label]} ({probabilities[predicted_label]:.2%})", fontsize=9)

            ax.set_xticks(range(num_classes))
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_box_aspect(1)
            ax.legend(handles=[plt.Line2D([0], [0], color="green", lw=4, label="True Label"),
                               plt.Line2D([0], [0], color="blue",  lw=4, label="Predicted Label")],
                      loc="upper right", fontsize=7)

    plt.suptitle("Predicted Images and Probability Distributions", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confusion(labels, predictions, class_names, model_name):
    """Print and plot raw counts with true classes on rows."""
    matrix = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Number of test images"})
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"{model_name}: Test Confusion Matrix")
    plt.setp(ax.get_yticklabels(), rotation=0)
    print(f"\n{model_name} Confusion Matrix:\n", matrix)
    plt.tight_layout()
    plt.show()


# ============================================================
# Helper Class for Model Training and Evaluation
# ============================================================
class ModelTrainer:
    """Train, evaluate, and visualize any Keras model.
    Classification (supervised / pretrained): pass class_names  -> confusion matrix, report, predicted images.
    Unsupervised (e.g. autoencoder)         : class_names=None, y = x, loss_function="mse", metrics=[].
    """

    def __init__(self, model, model_name, x_train, y_train, x_val, y_val, x_test, y_test, class_names=None,
                 optimizer="adam", loss_function="categorical_crossentropy", metrics=("accuracy",),
                 batch_size=batch_size, no_epochs=no_epochs, patience=patience, verbosity=2, display_images=None,
                 results_dir="results"):
        self.model          = model
        self.model_name     = model_name
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.class_names    = class_names
        self.optimizer      = optimizer
        self.loss_function  = loss_function
        self.metrics        = list(metrics)
        self.batch_size     = batch_size
        self.no_epochs      = no_epochs
        self.patience       = patience
        self.verbosity      = verbosity
        self.display_images = x_test if display_images is None else display_images     # Images shown in plots
        self.results_dir    = results_dir                                              # Where the best model is saved
        self.history        = None

    def train(self):
        """Compile and train the model with early stopping; keep the weights with the lowest validation loss."""
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        print("\n" + "=" * 60, f"Build, Train and Evaluate the {self.model_name} Model", "=" * 60, sep="\n")
        self.model.summary()

        os.makedirs(self.results_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.results_dir, f"{self.model_name.lower().replace(' ', '_')}_best.keras")
        callbacks = [ keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)]

        self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                                      epochs=self.no_epochs, batch_size=self.batch_size,
                                      callbacks=callbacks, verbose=self.verbosity)
        
        self.model = keras.models.load_model(checkpoint_path)                          # Best epoch, saved to disk
        return self.history

    def evaluate(self):
        """Evaluate the trained model on the test set and display its results."""
        val_loss = self.history.history["val_loss"]
        best     = int(np.argmin(val_loss))
        score    = self.model.evaluate(self.x_test, self.y_test, verbose=0, return_dict=True)

        # Training curves: accuracy if tracked, always loss
        if "accuracy" in self.history.history:
            plot_training_history(self.history, "accuracy", f"{self.model_name} Training vs Validation Accuracy")
        plot_training_history(self.history, "loss", f"{self.model_name} Training vs Validation Loss")

        print(f"\n{self.model_name} trained {len(val_loss)} epochs, best epoch {best + 1} (val loss {val_loss[best]:.4f})",
              *[f"test {k}: {v:.2%}" if k == "accuracy" else f"test {k}: {v:.4f}" for k, v in score.items()], sep=" | ")

        # Classification results (supervised / pretrained models only)
        predictions = self.model.predict(self.x_test, verbose=0)
        if self.class_names is not None:
            self.save_predictions(predictions)
            plot_confusion(self.y_test, predictions, self.class_names, self.model_name)
            print(f"\n{self.model_name} Classification Report:")
            print(classification_report(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1),target_names=self.class_names, digits=3))

            samples = np.linspace(0, len(self.x_test) - 1, 8).astype(int)            
            plot_image({"images": self.display_images[samples], "labels": self.y_test[samples]}, self.class_names, prediction_distribution=predictions[samples])
        return score, predictions

    def save_predictions(self, predictions):
        """Save the test probabilities to CSV"""
        true_idx, pred_idx = np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)
        df = pd.DataFrame({"true_label": np.array(self.class_names)[true_idx],
                           "predicted_label": np.array(self.class_names)[pred_idx],
                           "correct": true_idx == pred_idx})
        for i, name in enumerate(self.class_names):
            df[f"prob_{name}"] = predictions[:, i].round(6)                            # Softmax probability per class

        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"{self.model_name.lower().replace(' ', '_')}_test_predictions.csv")
        df.to_csv(path, index_label="index")
        print(f"Test predictions saved to {path}")
