
import os
import gc

import numpy as np
import pandas as pd

import pretrained_efficientnetb0_128 as baseline


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "efficientnetb0_scratch_frozen_bn"
)

CHECKPOINT_PATH = os.path.join(
    RESULTS_DIR,
    "efficientnetb0_scratch_frozen_bn_best.keras"
)


def run_evaluation():

    baseline.section(
        "Final Test Evaluation: EfficientNetB0 Scratch Frozen BN"
    )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    # Load only the held-out test dataset.
    test_images, test_labels = baseline.preprocess_dataset(
        baseline.testing_folder_path,
        size=baseline.image_size
    )

    x_test = test_images.astype(np.float32)

    y_test = baseline.to_categorical(
        test_labels,
        num_classes=len(baseline.classes)
    )

    del test_images
    gc.collect()

    print(f"\nTest images: {len(x_test)}")
    print(f"Image shape: {x_test.shape[1:]}")
    print(f"Data type: {x_test.dtype}")
    print(f"Pixel range: {x_test.min():.0f}–{x_test.max():.0f}")

    # Load the checkpoint selected by validation loss.
    model = baseline.keras.models.load_model(
        CHECKPOINT_PATH
    )

    print("\nLoaded checkpoint:")
    print(CHECKPOINT_PATH)

    # Evaluate using the same evaluation function as the baseline.
    metrics = baseline.evaluate_test(
        model,
        "EfficientNetB0_Scratch_Frozen_BN",
        x_test,
        y_test,
        RESULTS_DIR
    )

    # Add training information to the final comparison.
    history_df = pd.read_csv(
        os.path.join(RESULTS_DIR, "training_history.csv")
    )

    summary_df = pd.read_csv(
        os.path.join(RESULTS_DIR, "experiment_summary.csv")
    )

    metrics["Trainable Parameters"] = int(
        summary_df.loc[0, "Trainable Parameters"]
    )

    metrics["Epochs Run"] = len(history_df)

    metrics["Best Epoch"] = int(
        summary_df.loc[0, "Best Epoch"]
    )

    metrics["Training Time (min)"] = float(
        summary_df.loc[0, "Training Time (min)"]
    )

    metrics["Best Validation Accuracy"] = float(
        summary_df.loc[0, "Best Validation Accuracy"]
    )

    metrics["Best Validation Loss"] = float(
        summary_df.loc[0, "Best Validation Loss"]
    )

    metrics["BatchNormalization"] = "Frozen"

    output_path = os.path.join(
        RESULTS_DIR,
        "final_test_evaluation.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        output_path,
        index=False
    )

    print("\nFinal evaluation summary:")
    print(pd.DataFrame([metrics]).to_string(index=False))

    print(f"\nResults saved to: {RESULTS_DIR}")

    del model, x_test, y_test
    baseline.free_gpu_memory()


if __name__ == "__main__":
    run_evaluation()
