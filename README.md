# COMP263 – Brain Tumor MRI Classification (Group 5)

Classifying brain MRI images into 4 classes: glioma, meningioma, pituitary, no tumor.
We compare three approaches: supervised learning, unsupervised learning and state-of-the-art (SOTA) pretrained models.

## Group Members

| Member | Category | Model |
|---|---|---|
| Vaishnavi Aju | Supervised Learning | SVM |
| Lamia Alam Shova | Supervised Learning | CNN |
| Chitra Savaj | Unsupervised Learning | Autoencoder |
| Daniel Augusto Ramirez Duenas | Unsupervised Learning | Variational Autoencoder |
| Haripriya Gounalan | SOTA Models | ResNet50, data preprocessing |
| Abdallah Fardin Divkar | SOTA Models | EfficientNetB0, DenseNet121 |

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, Masoud Nickparvar):
7,200 images, 5,600 for training and 1,600 for testing. It is not included in this repository.

**Download:** on the Kaggle page click **Download**, then unzip into a `data` folder in the project root:

```
data/Training/{glioma, meningioma, notumor, pituitary}
data/Testing/{glioma, meningioma, notumor, pituitary}
```

**Processing:** done automatically by every script (`01_data_exploration_preprocessing/preprocessing.py`):
1. Crop each image to the brain and resize it (128x128 or 224x224)
2. Remove duplicate images (171 training, 16 test)
3. Scale pixels (divide by 255, or ImageNet preprocessing for pretrained models)
4. Split training images 80/20 into training (4,343) and validation (1,086); test set: 1,584 images

## Libraries

Python 3.13 with: `torch`, `keras` (PyTorch backend), `keras-tuner`, `numpy`, `pandas`, `matplotlib`, `seaborn`,
`scikit-learn`, `opencv-python-headless`, `pillow`. Exact versions are in `requirements.txt`.

```powershell
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128   # PyTorch with GPU support
pip install -r requirements.txt
```

A GPU is recommended. Pretrained models download their ImageNet weights on first run, so an internet connection is needed.

## How to Run

Run each script from the project root (the folder containing `data/`):

```powershell
python "01_data_exploration_preprocessing/data_exploration.py"
python "02_supervised learning/simple_cnn.py"
python "04_pretrained_model_finetuning/pretrained_resnet50_224.py"
```

| Folder | Scripts |
|---|---|
| `01_data_exploration_preprocessing/` | `data_exploration.py`, `preprocessing.py` |
| `02_supervised learning/` | `simple_cnn.py`, `cnn_tuning.py`, `svm.py` |
| `03_unsupervised_learning/` | `autoencoder.py` |
| `04_pretrained_model_finetuning/` | `model_screening.py`, `pretrained_resnet50_128.py`, `pretrained_resnet50_224.py`, `pretrained_resnet50_tuning_128.py`, `pretrained_resnet50_tuning_224.py`, EfficientNetB0 and DenseNet121 scripts |

Each script saves its results (logs, CSVs, plots) to a `results/` folder next to it.

**Save a log:** add `2>&1 | Tee-Object <log file>` (PowerShell) or `2>&1 | tee <log file>` (Linux/macOS).

**Long runs** (`cnn_tuning.py`, the ResNet50 224 and tuning scripts, several hours each) were run on a Kaggle GPU.
See [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for the Kaggle steps.

## Results

Every model was selected by its lowest validation loss. Train and validation accuracy are from that best epoch;
test results are on the 1,584 test images.

| Category | Best Model | Train Accuracy | Val Accuracy | Test Accuracy | Macro F1 |
|---|---|---|---|---|---|
| Supervised | Tuned CNN | 95.14% | 95.76% | 91.22% | 90.94% |
| Unsupervised | To be added | | | | |
| SOTA | ResNet50, fine-tuned and tuned (224x224) | 99.75% | 98.99% | **95.96%** | 95.86% |

## Best Models

| Model | Saved file | Script that creates it | Time (GPU) |
|---|---|---|---|
| Tuned CNN | `02_supervised learning/results/cnn_tuning/tuned_cnn_best.keras` (in git) | `02_supervised learning/cnn_tuning.py` | ~4 h |
| ResNet50, fine-tuned and tuned (224x224) | `04_pretrained_model_finetuning/results/pretrained_resnet50_tuning_224/resnet50_tuned_best.keras` (not in git, 95 MB) | `04_pretrained_model_finetuning/pretrained_resnet50_tuning_224.py` | ~7.5 h |

What the scripts need:
- `pretrained_resnet50_tuning_224.py` also needs `pretrained_resnet50_tuning_128.py` and `pretrained_resnet50_128.py` (same folder).
- Every script needs `01_data_exploration_preprocessing/data_exploration.py` (seed, class names, data paths),
  `01_data_exploration_preprocessing/preprocessing.py` (cropping, resizing, duplicate removal),
  `utils/model_utils.py` and the dataset in `data/`.
- The ResNet50 script needs an internet connection for the ImageNet weights.

To use a saved model, prepare images the same way as in training:
- Tuned CNN: 128x128, pixels divided by 255.
- ResNet50: 224x224, `keras.applications.resnet50.preprocess_input`.
- Output classes, in order: `glioma`, `meningioma`, `notumor`, `pituitary`.
