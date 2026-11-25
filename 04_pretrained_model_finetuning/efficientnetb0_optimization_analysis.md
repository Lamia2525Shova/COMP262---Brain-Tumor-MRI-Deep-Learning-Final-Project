# EfficientNetB0 Optimization Experiments: Consolidated Analysis

## Scope

Following the six-model baseline comparison, additional experiments investigated the validation instability observed in the EfficientNetB0 Scratch model. The optimization phase examined BatchNormalization (BN) behavior, global and selective BN freezing, and adaptive learning-rate scheduling.

This analysis consolidates the completed results. It does not require retraining or reevaluating any model.

## Consolidated results

| Configuration | BN configuration | Best validation accuracy | Best validation loss | Test accuracy | Test macro F1 | Training time |
|---|---|---:|---:|---:|---:|---:|
| Original Scratch | All 49 trainable | **96.41%** (epoch 28) | **0.1181** (epoch 28) | **92.55%** | **92.38%** | 24.76 min |
| Frozen BN | All 49 frozen | 91.34% (epoch 26) | 0.2498 (epoch 29) | 87.37% | 86.95% | 18.42 min |
| Selective BN | 4 frozen, 45 trainable | 91.90% (epoch 17) | 0.2619 (epoch 17) | Not evaluated | Not evaluated | 18.43 min |
| ReduceLROnPlateau | All 49 trainable | 95.95% (epoch 29) | 0.1419 (epoch 25) | 90.78% | 90.54% | 23.05 min |

**Checkpoint-selection note:** ReduceLROnPlateau's reported test metrics come from the checkpoint selected by minimum validation loss at epoch 25, not the epoch with the highest validation accuracy. Its highest validation accuracy occurred at epoch 29. The other validation-best values are reported separately for the same reason.

## Interpretation of the experiments

### 1. Original Scratch baseline

The original EfficientNetB0 Scratch model remains the strongest evaluated configuration. It achieved 92.55% test accuracy, 92.96% macro precision, 92.44% macro recall, and 92.38% macro F1. Its best validation accuracy and best validation loss were both recorded at epoch 28.

The training-history plots showed abrupt validation deterioration around epochs 5 and 17, including a validation-loss spike above 4 around epoch 17. This motivated further diagnostics. The unusual validation trajectory was a reason to investigate, but does not by itself invalidate the final test result.

### 2. BatchNormalization diagnostic

A separate diagnostic monitored the 49 BN layers during scratch training using training and validation data only. Moving statistics changed during training, and average moving variance increased from 1.4649 at epoch 3 to 2.5023 at epoch 8. Some deeper BN layers showed substantial changes.

These observations motivated BN-freezing experiments. However, changing moving statistics is expected during training and does not, on its own, demonstrate that BN was malfunctioning or caused the validation instability. The diagnostic establishes an association worth testing, not causation.

### 3. Global BN freezing

Freezing all 49 BN layers produced a comparatively smoother validation trajectory. All BN moving means and variances remained unchanged while convolutional weights continued to update.

Despite the smoother trajectory, performance was lower: test accuracy declined from 92.55% to 87.37%, and macro F1 declined from 92.38% to 86.95%. The result indicates that global BN freezing was not an effective intervention for improving generalization in this experiment.

### 4. Selective BN freezing

The selective experiment froze four deeper BN layers—`block6b_project_bn`, `block6c_project_bn`, `block6d_project_bn`, and `top_bn`—while leaving the other 45 BN layers trainable. It reached 91.90% best validation accuracy, modestly above the globally frozen experiment's 91.34%.

However, its best validation loss was 0.2619, slightly higher than the globally frozen experiment's 0.2498. Both remained below the original Scratch baseline's validation performance. This was retained as a validation-only diagnostic and was not evaluated on the test set. No claim about its test performance should be made.

### 5. ReduceLROnPlateau

The learning-rate scheduling experiment kept all BN layers trainable and reduced the learning rate when validation loss stopped improving. It achieved 95.95% best validation accuracy and 0.1419 best validation loss.

The selected validation-loss checkpoint achieved 90.78% test accuracy, 91.31% macro precision, 90.65% macro recall, and 90.54% macro F1. This was stronger than either BN-freezing configuration with available test results, but it did not surpass the original Scratch model.

At class level, ReduceLROnPlateau's glioma recall was 76.17%, compared with 79.79% for the original Scratch model. It also classified 48 glioma images as no tumor, compared with 40 for the original model. Therefore, the learning-rate schedule did not resolve the principal glioma classification difficulty.

## Overall conclusion and model selection

The optimization experiments show that a smoother validation trajectory or improved validation metrics do not necessarily translate into stronger held-out test performance.

The original EfficientNetB0 Scratch model remains the selected EfficientNetB0 configuration because it achieved the strongest test accuracy and macro F1 among the configurations evaluated on the test set. ReduceLROnPlateau is a useful comparison because it improved validation behavior relative to the BN-freezing experiments, but its test performance remained below the original baseline. Selective BN freezing remains a validation-only diagnostic and should not be ranked by test performance.

The strongest pretrained EfficientNetB0 configuration from the original baseline phase remains Fine-tuned (83.46% test accuracy and 82.72% macro F1). The scratch model's result is specific to this dataset, preprocessing pipeline, split, and training setup; it should not be generalized into a claim that training from scratch is universally better than ImageNet pretraining.

## Methodological limitations

- The experiments use a single random seed and one train-validation split; results do not establish statistical significance.
- The same held-out test set was used for the baseline comparisons and subsequent candidate evaluations. The later evaluations are comparisons of validation-selected candidates, not an untouched independent test of the entire optimization process.
- Selective BN freezing was not evaluated on the test set; its generalization performance is unknown.
- Validation stability is assessed from the recorded training histories and is not a formal measure of robustness across repeated runs.

## Decision

Retain the original EfficientNetB0 Scratch checkpoint as the selected EfficientNetB0 model. Preserve Frozen BN, Selective BN, and ReduceLROnPlateau as completed diagnostic/optimization experiments. Do not rerun these experiments merely to regenerate comparison tables or charts; use the recorded metrics and histories.
