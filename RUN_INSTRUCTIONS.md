# How to Run the Project

## 1. Setup

1. Install the packages (Python 3.13):
   ```powershell
   pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128   # PyTorch with GPU support
   pip install -r requirements.txt
   ```
2. Download the dataset from Kaggle (`masoudnickparvar/brain-tumor-mri-dataset`) and unzip it so the project has:
   ```
   data/Training/{glioma, meningioma, notumor, pituitary}
   data/Testing/{glioma, meningioma, notumor, pituitary}
   ```
3. Open a terminal in the project folder (the one that contains `data/` and `utils/`).
   Every script finds the project folder itself, so the commands below work from there.

## 2. Scripts

| # | Script | Notes |
|---|---|---|
| 1 | `01_data_exploration_preprocessing/data_exploration.py` | Class counts, image sizes, duplicates, sample images |
| 2 | `01_data_exploration_preprocessing/preprocessing.py` | Crop, resize, duplicate removal |
| 3 | `02_supervised learning/simple_cnn.py` | Baseline CNN (2 conv layers) |
| 4 | `02_supervised learning/cnn_tuning.py` | Hyperband search, then evaluates the best trial (~4 h) |
| 5 | `04_pretrained_model_finetuning/model_screening.py` | Compares 4 ImageNet models on validation data |
| 6 | `04_pretrained_model_finetuning/pretrained_resnet50_128.py` | ResNet50 frozen, fine-tuned and from scratch at 128x128 |
| 7 | `04_pretrained_model_finetuning/pretrained_resnet50_224.py` | Same at 224x224 (~1 h) |
| 8 | `04_pretrained_model_finetuning/pretrained_resnet50_tuning_128.py` | Random search of the ResNet50 fine-tuning stage, 20 trials (~5 h) |
| 9 | `04_pretrained_model_finetuning/pretrained_resnet50_tuning_224.py` | Same at 224x224 (~7.5 h) |

- Every script loads and preprocesses the data itself, so each one can be run on its own. Scripts 1 and 2 are optional.
- Scripts 5 to 9 download ImageNet weights, so they need an internet connection.
- Results (logs, CSVs, plots, predictions) are saved to a `results/` folder next to each script.
- A finished KerasTuner search (scripts 4, 8, 9) is reloaded instead of rerun if its trial folder exists.
- Times are on a Kaggle GPU. Scripts 4, 7, 8 and 9 are best run on Kaggle (section 5).

## 3. Run without a log

Output only appears in the terminal (or run the file in Spyder / VS Code with the Run button):

```powershell
python "01_data_exploration_preprocessing/data_exploration.py"
python "01_data_exploration_preprocessing/preprocessing.py"
python "02_supervised learning/simple_cnn.py"
python "02_supervised learning/cnn_tuning.py"
python "04_pretrained_model_finetuning/model_screening.py"
python "04_pretrained_model_finetuning/pretrained_resnet50_128.py"
python "04_pretrained_model_finetuning/pretrained_resnet50_224.py"
python "04_pretrained_model_finetuning/pretrained_resnet50_tuning_128.py"
python "04_pretrained_model_finetuning/pretrained_resnet50_tuning_224.py"
```

## 4. Run with a log (PowerShell 7)

The output is shown **and** saved to a `_log.txt` file in the script's results folder.
Run these two lines once per terminal first:

```powershell
$env:PYTHONIOENCODING = "utf-8"     # log text is saved as UTF-8
$env:MPLBACKEND = "Agg"             # optional: no plot windows, so long runs never wait for a window to be closed
```

Then run any script:

```powershell
python -u "01_data_exploration_preprocessing/data_exploration.py" 2>&1 | Tee-Object "01_data_exploration_preprocessing/results/data_exploration/data_exploration_log.txt"
python -u "01_data_exploration_preprocessing/preprocessing.py" 2>&1 | Tee-Object "01_data_exploration_preprocessing/results/preprocessing/preprocessing_log.txt"
python -u "02_supervised learning/simple_cnn.py" 2>&1 | Tee-Object "02_supervised learning/results/simple_cnn/simple_cnn_log.txt"
python -u "02_supervised learning/cnn_tuning.py" 2>&1 | Tee-Object "02_supervised learning/results/cnn_tuning/cnn_tuning_log.txt"
python -u "04_pretrained_model_finetuning/model_screening.py" 2>&1 | Tee-Object "04_pretrained_model_finetuning/results/model_screening/model_screening_log.txt"
python -u "04_pretrained_model_finetuning/pretrained_resnet50_128.py" 2>&1 | Tee-Object "04_pretrained_model_finetuning/results/pretrained_resnet50_128/resnet50_128_log.txt"
python -u "04_pretrained_model_finetuning/pretrained_resnet50_224.py" 2>&1 | Tee-Object "04_pretrained_model_finetuning/results/pretrained_resnet50_224/resnet50_224_log.txt"
python -u "04_pretrained_model_finetuning/pretrained_resnet50_tuning_128.py" 2>&1 | Tee-Object "04_pretrained_model_finetuning/results/pretrained_resnet50_tuning_128/resnet50_tuning_128_log.txt"
python -u "04_pretrained_model_finetuning/pretrained_resnet50_tuning_224.py" 2>&1 | Tee-Object "04_pretrained_model_finetuning/results/pretrained_resnet50_tuning_224/resnet50_tuning_224_log.txt"
```

Notes:
- `-u` prints each epoch as it finishes instead of in chunks.
- `2>&1` also saves warnings and errors, so a crash is visible in the log.
- Use **PowerShell 7** (`pwsh`). Windows PowerShell 5.1 saves `Tee-Object` files as UTF-16, which some editors show as garbled text.
- `Tee-Object` does not create folders. On a fresh copy, create the results folder first, e.g.
  `New-Item -ItemType Directory -Force "02_supervised learning/results/simple_cnn"`.
- With `MPLBACKEND = "Agg"`, the model scripts still save their plots as PNG files, but the
  data exploration and preprocessing plots are only displayed, so run those two without it.
- Git Bash / Linux / macOS: same commands with `| tee <log file>` instead of `| Tee-Object <log file>`.

## 5. Run on Kaggle (GPU)

1. Zip the scripts (all `.py` files in the numbered folders, plus `utils/model_utils.py`) and upload the zip
   as a Kaggle dataset. Add it and `masoudnickparvar/brain-tumor-mri-dataset` to a notebook.
2. Settings: **Accelerator GPU T4 or P100**, **Internet on** (for the ImageNet weights).
3. First cell: copy the scripts into one folder and link the data:
   ```python
   import glob, os, shutil
   os.chdir("/kaggle/working")
   for f in glob.glob("/kaggle/input/**/*.py", recursive=True):
       shutil.copy(f, ".")
   data_dir = glob.glob("/kaggle/input/**/brain-tumor-mri-dataset", recursive=True)[0]
   if os.path.islink("data"):
       os.unlink("data")
   os.symlink(data_dir, "data")
   ```
4. One cell per script. Without a log: `!python -u cnn_tuning.py`. With a log:
   ```bash
   !python -u cnn_tuning.py 2>&1 | tee cnn_tuning_log.txt
   !python -u model_screening.py 2>&1 | tee model_screening_log.txt
   !python -u pretrained_resnet50_224.py 2>&1 | tee resnet50_224_log.txt
   !python -u pretrained_resnet50_tuning_128.py 2>&1 | tee resnet50_tuning_128_log.txt
   !python -u pretrained_resnet50_tuning_224.py 2>&1 | tee resnet50_tuning_224_log.txt
   ```
   Run the two tuning scripts in **separate notebooks**: each gets its own GPU and its own 12-hour limit.
5. **Save Version → Save & Run All (Commit)**. When it finishes, download the `results` folder
   and the log files from the version's **Output** tab. Leave out the `hyperband_*` / `random_search_*`
   trial folders: they are several GB and are not kept in git.
