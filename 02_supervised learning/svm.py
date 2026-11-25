# Vaishnavi Aju
# SVM for brain tumor MRI

import json
import os
import sys
import time
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# project paths
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _here if os.path.exists(os.path.join(_here, "model_utils.py")) else os.path.dirname(_here)
sys.path[:0] = [os.path.join(PROJECT_ROOT, d) for d in ("01_data_exploration_preprocessing", "utils")]
os.chdir(PROJECT_ROOT)

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.utils import to_categorical
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from data_exploration import SEED, classes, section, show, testing_folder_path, training_folder_path
from model_utils import plot_confusion, plot_image, show_and_save
from preprocessing import preprocess_dataset

# settings
results_dir  = os.path.join(_here, "results", "svm")
pca_variance = 0.95
default_svm  = {"kernel": "rbf", "C": 10, "gamma": "scale"}

C_values     = [0.1, 1, 10, 100]
gamma_values = [0.3, 1, 3]  # x scale

scalers = {"none": None, "min-max": MinMaxScaler, "standard": StandardScaler}

cnn_predictions = {"Simple CNN": "simple_cnn/simple_cnn_test_predictions.csv",
                   "Tuned CNN":  "cnn_tuning/tuned_cnn_test_predictions.csv"}


# image -> feature vector
class FeatureExtractor:

    methods = ("raw", "hog", "lbp", "combined")

    def __init__(self, method, hog_cell=16, hog_bins=9, grid=4):
        if method not in self.methods:
            raise ValueError(f"method must be one of {self.methods}, not {method!r}")
        self.method   = method
        self.hog_cell = hog_cell
        self.hog_bins = hog_bins
        self.grid     = grid

    def extract(self, images):
        function = getattr(self, f"_{self.method}")
        return np.stack([function(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) for img in images]).astype(np.float32)

    def _raw(self, gray):
        return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).ravel() / 255

    def _hog(self, gray):
        pixels = np.sqrt(gray.astype(np.float32) / 255)  # gamma correction
        gx = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # vote into 2 nearest bins
        position = (angle % 180) / (180 / self.hog_bins)
        low      = np.floor(position).astype(int)
        share    = position - low
        rows, cols = gray.shape[0] // self.hog_cell, gray.shape[1] // self.hog_cell
        y, x  = np.indices(gray.shape) // self.hog_cell
        cell  = y * cols + x
        cells = sum(np.bincount((cell * self.hog_bins + b % self.hog_bins).ravel(), weights=w.ravel(),
                                minlength=rows * cols * self.hog_bins)
                    for b, w in ((low, magnitude * (1 - share)), (low + 1, magnitude * share)))
        cells = cells.reshape(rows, cols, self.hog_bins)

        # normalize 2x2 blocks
        blocks = np.concatenate([cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]], axis=-1)
        blocks = blocks / (np.linalg.norm(blocks, axis=-1, keepdims=True) + 1e-6)
        blocks = np.minimum(blocks, 0.2)  # clip strong edges
        return (blocks / (np.linalg.norm(blocks, axis=-1, keepdims=True) + 1e-6)).ravel()

    def _regions(self, array):
        return [part for row in np.array_split(array, self.grid) for part in np.array_split(row, self.grid, axis=1)]

    def _lbp(self, gray):
        center    = gray[1:-1, 1:-1]
        h, w      = gray.shape
        offsets   = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        neighbors = np.stack([gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] >= center for dy, dx in offsets])
        changes   = np.sum(neighbors != np.roll(neighbors, 1, axis=0), axis=0)
        codes     = np.where(changes <= 2, neighbors.sum(axis=0), 9)  # 9 = non-uniform
        return np.concatenate([np.bincount(r.ravel(), minlength=10) / r.size for r in self._regions(codes)])

    def _intensity(self, gray):
        return np.concatenate([np.bincount(r.ravel() // 16, minlength=16) / r.size for r in self._regions(gray)])

    def _combined(self, gray):
        return np.concatenate([self._hog(gray), self._lbp(gray), self._intensity(gray)])


# train, tune, test
class SVMExperiment:

    def __init__(self, train, val, test, class_names, results_dir):
        self.images = {"train": train[0], "val": val[0], "test": test[0]}
        self.labels = {"train": train[1], "val": val[1], "test": test[1]}
        self.class_names = class_names
        self.results_dir = results_dir
        self.cache       = {}
        self.best        = {"method": None, "scaler": "standard", "pca": True, "flip": False, "svm": dict(default_svm)}
        self.model       = None
        os.makedirs(results_dir, exist_ok=True)

    def path(self, filename):
        return os.path.join(self.results_dir, filename)

    def features(self, split, method, flip=False):
        key = (split, method, flip)
        if key not in self.cache:
            images = self.images[split][:, :, ::-1] if flip else self.images[split]
            self.cache[key] = FeatureExtractor(method).extract(np.ascontiguousarray(images))
        return self.cache[key]

    def training_data(self, method, flip):
        x, y = self.features("train", method), self.labels["train"]
        # train only
        if flip:
            x, y = np.concatenate([x, self.features("train", method, flip=True)]), np.concatenate([y, y])
        return x, y

    @staticmethod
    def transform_steps(scaler, pca):
        steps = [scalers[scaler]()] if scalers[scaler] else []
        if pca:
            steps.append(PCA(n_components=pca_variance, svd_solver="full", random_state=SEED))
        return steps or ["passthrough"]

    @staticmethod
    def build_svm(svm):
        return SVC(**svm, class_weight="balanced", cache_size=1000)

    @staticmethod
    def scores(y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": precision, "Recall": recall, "Macro F1": f1}

    def validate(self, method, scaler, pca, flip, svm):
        x, y      = self.training_data(method, flip)
        x_val     = self.features("val", method)
        start     = time.perf_counter()
        transform = make_pipeline(*self.transform_steps(scaler, pca))
        x         = transform.fit_transform(x)
        model     = self.build_svm(svm).fit(x, y)
        seconds   = time.perf_counter() - start
        y_pred    = model.predict(transform.transform(x_val))
        return {"SVM Inputs": x.shape[1], "Support Vectors": int(model.n_support_.sum()),
                "Val_Accuracy": accuracy_score(self.labels["val"], y_pred),
                "Val_Macro_F1": f1_score(self.labels["val"], y_pred, average="macro"), "Train Seconds": round(seconds, 2)}

    def report(self, title, rows, filename):
        df = pd.DataFrame(rows)
        df["Val_Macro_F1"] = df["Val_Macro_F1"].round(6)
        df = df.sort_values(["Val_Macro_F1", "SVM Inputs"], ascending=[False, True], kind="stable").reset_index(drop=True)
        df.to_csv(self.path(filename), index=False)
        show(title, df.round(4))
        return df

    def plot_bars(self, df, labels, title, filename):
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.5))
        bars = ax.barh(labels[::-1], df["Val_Macro_F1"][::-1], color="#2A9D8F")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(xlabel="Validation macro F1", title=title, xlim=(0, 1.08))
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        show_and_save(self.path(filename))

    # step 3
    def compare_features(self):
        rows = [{"Feature Set": method, "Extracted Features": self.features("train", method).shape[1],
                 **self.validate(**{**self.best, "method": method})} for method in FeatureExtractor.methods]
        df = self.report("Feature sets (standard scaling, PCA 95%, RBF SVM, C = 10)", rows, "feature_comparison.csv")
        self.plot_bars(df, df["Feature Set"].tolist(), "SVM: Feature Extraction Methods", "svm_feature_comparison.png")
        self.best["method"] = df["Feature Set"][0]
        print(f"\n  Selected features: {self.best['method']}")

    # step 4
    def compare_normalization(self):
        rows = [{"Scaler": scaler, "PCA": pca, "Flip Augmentation": flip,
                 **self.validate(**{**self.best, "scaler": scaler, "pca": pca, "flip": flip})}
                for scaler in scalers for pca in (False, True) for flip in (False, True)]
        df = self.report("Normalization, PCA and augmentation (RBF SVM, C = 10)", rows, "normalization_comparison.csv")
        labels = [s + (" + PCA" if p else "") + (" + flip" if f else "")
                  for s, p, f in zip(df["Scaler"], df["PCA"], df["Flip Augmentation"])]
        self.plot_bars(df, labels, "SVM: Normalization, PCA and Augmentation", "svm_normalization_comparison.png")
        self.best.update(scaler=df["Scaler"][0], pca=bool(df["PCA"][0]), flip=bool(df["Flip Augmentation"][0]))
        print(f"\n  Selected: {labels[0]}")

    # step 5
    def tune(self):
        x, _  = self.training_data(self.best["method"], self.best["flip"])
        x     = make_pipeline(*self.transform_steps(self.best["scaler"], self.best["pca"])).fit_transform(x)
        # same as gamma="scale"
        scale = float(1 / (x.shape[1] * x.var()))

        grid  = [{"kernel": "linear", "C": C} for C in C_values]
        grid += [{"kernel": "poly", "degree": d, "C": C, "gamma": scale} for d in (2, 3) for C in C_values]
        grid += [{"kernel": "rbf", "C": C, "gamma": scale * g} for C in C_values for g in gamma_values]

        rows = []
        for i, svm in enumerate(grid, 1):
            row = {**svm, "gamma / scale": svm["gamma"] / scale if "gamma" in svm else np.nan,
                   **self.validate(**{**self.best, "svm": svm})}
            rows.append(row)
            print(f"    [{i}/{len(grid)}] {svm} -> val macro F1 {row['Val_Macro_F1']:.4f} ({row['Train Seconds']:.1f}s)")
        df = self.report("Hyperparameter search (best first)", rows, "hyperparameter_search.csv")

        winner = df.iloc[0]
        self.best["svm"] = {"kernel": winner["kernel"], "C": float(winner["C"])}
        if winner["kernel"] != "linear":
            self.best["svm"]["gamma"] = float(winner["gamma"])
        if winner["kernel"] == "poly":
            self.best["svm"]["degree"] = int(winner["degree"])
        with open(self.path("best_hyperparameters.json"), "w", encoding="utf-8") as file:
            json.dump(self.best, file, indent=2)
        print(f"\n  Selected: {self.best['svm']}")
        self.plot_tuning(df)

    def plot_tuning(self, df):
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
        names = df["kernel"] + df["degree"].map(lambda d: "" if pd.isna(d) else f" (degree {d:.0f})")
        for name, group in df.groupby(names):
            curve = group.groupby("C")["Val_Macro_F1"].max()
            left.plot(curve.index, curve.values, "o-", label=name)
        left.set(xscale="log", xlabel="C (larger = less regularization)", ylabel="Validation macro F1", title="Kernel and C")
        left.legend()
        left.grid(alpha=0.3)
        rbf = df[df["kernel"] == "rbf"].assign(g=lambda d: d["gamma / scale"].round(2))
        sns.heatmap(rbf.pivot(index="C", columns="g", values="Val_Macro_F1"), annot=True, fmt=".3f", cmap="Blues", ax=right)
        right.set(xlabel="gamma (x scale)", title="RBF kernel: C and gamma (validation macro F1)")
        plt.tight_layout()
        show_and_save(self.path("svm_hyperparameter_search.png"))

    # step 6
    def evaluate(self):
        best   = self.best
        method = best["method"]
        show("Selected configuration", pd.Series({**{k: v for k, v in best.items() if k != "svm"}, **best["svm"]},
                                                 name="Value").to_frame())
        x, y  = self.training_data(method, best["flip"])
        start = time.perf_counter()
        self.model = make_pipeline(*self.transform_steps(best["scaler"], best["pca"]), self.build_svm(best["svm"])).fit(x, y)
        seconds = time.perf_counter() - start
        joblib.dump({"pipeline": self.model, "method": method, "classes": self.class_names},
                    self.path("svm_best.joblib"), compress=3)

        splits = pd.DataFrame([{"Split": split, **self.scores(self.labels[split], self.model.predict(self.features(split, method)))}
                               for split in ("train", "val", "test")]).set_index("Split")
        splits.to_csv(self.path("svm_scores.csv"))
        show("SVM scores per split", splits.round(4))
        print(f"\nSVM training time: {seconds:.1f} s | support vectors: {self.model[-1].n_support_.sum()}"
              f" of {len(x)} training vectors | test accuracy: {splits.loc['test', 'Accuracy']:.2%}")

        x_test, y_test = self.features("test", method), self.labels["test"]
        y_pred = self.model.predict(x_test)

        # scores -> probabilities
        proba = (CalibratedClassifierCV(FrozenEstimator(self.model), method="sigmoid")
                 .fit(self.features("val", method), self.labels["val"]).predict_proba(x_test))

        num_classes = len(self.class_names)
        plot_confusion(to_categorical(y_test, num_classes), to_categorical(y_pred, num_classes), self.class_names,
                       "SVM", self.path("svm_confusion_matrix.png"))
        print("\nSVM Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=3))

        samples = np.linspace(0, len(y_test) - 1, 8).astype(int)
        plot_image({"images": self.images["test"][samples], "labels": y_test[samples]}, self.class_names,
                   prediction_distribution=proba[samples], save_path=self.path("svm_sample_predictions.png"))

        names = np.array(self.class_names)
        df = pd.DataFrame({"true_label": names[y_test], "predicted_label": names[y_pred], "correct": y_test == y_pred})
        for i, name in enumerate(self.class_names):
            df[f"prob_{name}"] = proba[:, i].round(6)
        df.to_csv(self.path("svm_test_predictions.csv"), index_label="index")
        return df

    # step 7
    def compare_with_cnns(self, svm_predictions):
        models = {"SVM": svm_predictions}
        for name, file in cnn_predictions.items():
            path = os.path.join(_here, "results", file)
            if not os.path.exists(path):
                print(f"  {name} skipped: {path} not found")
                continue
            df = pd.read_csv(path)
            if not df["true_label"].equals(svm_predictions["true_label"]):
                print(f"  {name} skipped: its test images are not the same as the SVM's")
                continue
            models[name] = df

        rows, recalls = [], {}
        for name, df in models.items():
            ours, theirs = svm_predictions["correct"], df["true_label"] == df["predicted_label"]
            row = {"Model": name, **self.scores(df["true_label"], df["predicted_label"])}
            if name != "SVM":
                svm_only, other_only = int((ours & ~theirs).sum()), int((~ours & theirs).sum())
                row.update({"Only SVM correct": svm_only, "Only this model correct": other_only,
                            "McNemar p": binomtest(svm_only, svm_only + other_only).pvalue if svm_only + other_only else 1.0})
            rows.append(row)
            recalls[name] = theirs.groupby(df["true_label"]).mean()
        table, recalls = pd.DataFrame(rows).set_index("Model"), pd.DataFrame(recalls)
        table.to_csv(self.path("svm_vs_cnn.csv"))
        show("SVM vs CNN on the test set", table.round(4).T)
        show("Recall per class", recalls.round(3))

        fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5))
        metrics = table[["Accuracy", "Precision", "Recall", "Macro F1"]].T
        for ax, data, title in ((left, metrics, "Test metrics"), (right, recalls, "Recall per class")):
            data.plot.bar(ax=ax, rot=0, width=0.8, color=["#2A9D8F", "#F4A261", "#264653"][:data.shape[1]])
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
            ax.set(title=title, xlabel="", ylim=(0, 1.12))
            ax.legend(loc="lower right")
        plt.tight_layout()
        show_and_save(self.path("svm_vs_cnn.png"))


if __name__ == "__main__":
    section("Step 1: Load and Preprocess the Dataset")
    train_images, train_labels = preprocess_dataset(training_folder_path)
    test_images,  test_labels  = preprocess_dataset(testing_folder_path)

    section("Step 2: Split Training Data into Training and Validation")
    # same split as the CNNs
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.20,
                                                      stratify=train_labels, random_state=SEED)
    show("Dataset splits", pd.DataFrame({"Images": [len(x_train), len(x_val), len(test_images)]},
                                        index=["Training", "Validation", "Testing"]))

    experiment = SVMExperiment((x_train, y_train), (x_val, y_val), (test_images, test_labels), classes, results_dir)

    section("Step 3: Compare Feature Extraction Methods")
    experiment.compare_features()

    section("Step 4: Compare Normalization, PCA and Augmentation")
    experiment.compare_normalization()

    section("Step 5: Tune the SVM Hyperparameters")
    experiment.tune()

    section("Step 6: Evaluate the Best SVM on the Test Set")
    predictions = experiment.evaluate()

    section("Step 7: Compare with the CNNs")
    experiment.compare_with_cnns(predictions)
