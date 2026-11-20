# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 2026

@author: Haripriya Gounalan
Runs the same ResNet50 fine-tuning search as pretrained_resnet50_tuning_128.py, but at 224x224 (ResNet50's ImageNet size).
Results go to results/pretrained_resnet50_tuning_224/.
Batch size 16 for Stage 1 and 8 or 16 for fine-tuning (instead of 32 / 16 or 32): 224x224 images need about 3x the GPU memory of 128x128.
"""

from pretrained_resnet50_tuning_128 import run_tuning

if __name__ == "__main__":
    run_tuning((224, 224), batch_size=16, batch_sizes=[8, 16])
