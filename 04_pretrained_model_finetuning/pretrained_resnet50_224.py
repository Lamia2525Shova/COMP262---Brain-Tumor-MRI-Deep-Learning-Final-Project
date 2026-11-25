# -*- coding: utf-8 -*-
"""
Created on Sat Nov 7 2026

@author: Haripriya Gounalan
Runs the same ResNet50 experiment as pretrained_resnet50_128.py, but at 224x224 (ResNet50's ImageNet size).
Results go to results/pretrained_resnet50_224/.
Batch size 16 (instead of 32): 224x224 images need about 3x the GPU memory of 128x128.
"""

from pretrained_resnet50_128 import run_experiment

if __name__ == "__main__":
    run_experiment((224, 224), batch_size=16)
