# Baseline Experimental Findings

### **1. Best-Performing Architecture**

Among the six baseline experiments, **EfficientNetB0 trained from scratch achieved the strongest overall test performance**, with an accuracy of **92.55%** and a macro F1-score of **92.38%**. It also achieved the highest macro precision (92.96%) and macro recall (92.44%) among the evaluated configurations.

DenseNet121 performed best when fine-tuned from ImageNet-pretrained weights, achieving 88.57% test accuracy and an 88.26% macro F1-score. Although DenseNet121 fine-tuning outperformed EfficientNetB0 fine-tuning, it did not surpass EfficientNetB0 trained from scratch.

These findings suggest that, under the selected experimental conditions, EfficientNetB0 was the stronger overall architecture. However, this conclusion is specific to the dataset, input resolution, training configuration, and single-seed experiments used in this project.

### **2. Best-Performing Initialization Strategy**

The best initialization strategy differed between the two architectures:

- **EfficientNetB0:** Random initialization (training from scratch) achieved the best results, with 92.55% accuracy and 92.38% macro F1.
- **DenseNet121:** ImageNet-pretrained weights followed by fine-tuning achieved the best results, with 88.57% accuracy and 88.26% macro F1.

For EfficientNetB0, training from scratch outperformed its fine-tuned counterpart by approximately **9.09 percentage points in accuracy** and **9.66 percentage points in macro F1**. For DenseNet121, fine-tuning improved accuracy by approximately **5.62 percentage points** and macro F1 by **5.54 percentage points** compared with its frozen-backbone experiment.

Therefore, training from scratch was more effective for EfficientNetB0 in this particular setup, whereas fine-tuning pretrained weights was more effective for DenseNet121.

### **3. Effect of ImageNet Pretraining**

ImageNet pretraining did not produce consistent improvements across both architectures.

For EfficientNetB0, the frozen and fine-tuned pretrained configurations achieved lower test performance than the scratch configuration. The frozen model achieved 80.56% accuracy, while fine-tuning improved it to 83.46%. However, both remained below the 92.55% accuracy obtained by training from scratch.

In contrast, DenseNet121 benefited from ImageNet pretraining. Fine-tuning improved test accuracy from 82.95% with a frozen backbone to 88.57%, while macro F1 increased from 82.71% to 88.26%. This indicates that allowing selected deeper layers to adapt to the MRI dataset was more effective than using the pretrained feature extractor without modification.

The results demonstrate that the effectiveness of transfer learning depends on the architecture and experimental configuration. Features learned from ImageNet may provide a useful starting point, but they do not necessarily outperform training from scratch when applied to a specialized medical-imaging dataset.

### **4. Training Efficiency**

Training efficiency was evaluated using the number of trainable parameters, epochs completed, and training time.

| Architecture | Strategy | Trainable Parameters | Best Epoch | Training Time |
| --- | --- | --- | --- | --- |
| EfficientNetB0 | Frozen | 5,124 | 10 | 5.30 min |
| EfficientNetB0 | Fine-tuned | 1,126,708 | 18 | 12.00 min |
| EfficientNetB0 | Scratch | 4,012,672 | 28 | 24.76 min |
| DenseNet121 | Frozen | 4,100 | 7 | 8.15 min |
| DenseNet121 | Fine-tuned | 2,134,020 | 19 | 19.08 min |
| DenseNet121 | Scratch | 6,957,956 | 16 | 51.49 min |

The frozen-backbone experiments were the most computationally efficient because only the classification head was trained. Fine-tuning required more time because additional backbone layers were updated.

Among the six experiments, EfficientNetB0 trained from scratch offered a particularly strong balance between performance and training time. It achieved the highest accuracy and macro F1 while requiring approximately half the training time of DenseNet121 trained from scratch.

DenseNet121 trained from scratch was the least efficient configuration, taking 51.49 minutes while achieving 88.32% accuracy. This was approximately 27 minutes longer than EfficientNetB0 trained from scratch, despite producing lower test performance.

Overall, EfficientNetB0 provided the most favourable combination of predictive performance and computational efficiency in these baseline experiments.

### **5. Limitations and Potential Improvements**

Although the baseline experiments provide a useful comparison between the two architectures, several limitations should be considered.

**1. Limited experimental repetitions**

Each configuration was evaluated using a single random seed. Consequently, the results may be affected by random initialization, data splitting, and training variability. Repeating experiments with multiple seeds would provide a more reliable comparison and allow reporting mean performance and standard deviation.

**2. Fixed input resolution**

All experiments used 128 × 128 images. Although this resolution reduces computational requirements, it may remove fine-grained visual information relevant to distinguishing between tumour categories. Experiments with higher resolutions, such as 224 × 224, could determine whether additional image detail improves classification performance.

**3. Limited hyperparameter optimization**

The baseline experiments used predefined learning rates, dropout, batch size, and early-stopping settings. These parameters were not extensively optimized for each architecture. Further experimentation with learning rates, dropout, batch sizes, and fine-tuning depth may improve performance.

**4. Class-specific classification errors**

The confusion matrices revealed that glioma and meningioma were more challenging to distinguish than the other classes. EfficientNetB0 trained from scratch achieved the highest glioma recall at 79.79%, but still misclassified 78 of the 386 glioma test images. Notably, 40 of these were classified as no tumour.

This suggests that overall accuracy alone does not fully represent classification performance. Future optimization should prioritize improving glioma recall while maintaining performance across the remaining classes. Potential approaches include class-weighted loss functions, carefully selected data augmentation, and targeted learning-rate adjustments.

**5. Limited data augmentation investigation**

The baseline comparison does not establish whether alternative augmentation strategies could improve generalization. Further experiments could investigate realistic MRI transformations, such as small rotations, translations, zoom adjustments, and contrast variations. These transformations should preserve medically relevant image characteristics.

**6. Dataset and evaluation limitations**

The experiments used one publicly available dataset with a fixed training and testing split. Although duplicate images were removed within each split and no train-test duplicates were identified, this does not establish that the dataset is free from other sources of bias or distribution differences. Evaluation on an independent MRI dataset would provide stronger evidence of generalization.

Additionally, the test set was used to compare the baseline configurations. Future optimization should use validation performance for model selection and reserve an independent test evaluation for the final selected model.

### **6. Overall Conclusion**

The baseline comparison identified **EfficientNetB0 trained from scratch as the strongest overall configuration**, achieving 92.55% test accuracy and 92.38% macro F1. DenseNet121 fine-tuned from ImageNet weights was the strongest transfer-learning configuration, achieving 88.57% accuracy and 88.26% macro F1.

The experiments also demonstrated that ImageNet pretraining did not provide a universal advantage: DenseNet121 benefited substantially from fine-tuning, while EfficientNetB0 performed better when trained from scratch under the selected settings.

Considering predictive performance and training time together, EfficientNetB0 trained from scratch is the primary candidate for further optimization. The next stage should focus particularly on improving glioma classification, investigating augmentation and hyperparameter configurations, and evaluating whether the observed improvements remain consistent across repeated experiments.