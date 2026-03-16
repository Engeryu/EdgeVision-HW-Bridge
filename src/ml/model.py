# ===========================================================
#  File    : model.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-16
# ===========================================================

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet18, resnet50

from src.config import cfg
from src.ml.dataset import get_num_classes


class SimpleCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network for image classification.

    Designed as the primary hardware-bridge target: its first convolutional
    layer (conv1) is used by the Amaranth MAC testbench for weight extraction,
    quantization, and cycle-accurate co-simulation. This architecture is
    intentionally kept shallow to remain deployable on edge ASIC/FPGA targets.
    """

    def __init__(self, num_classes: int = 0):
        super().__init__()
        num_classes = num_classes or get_num_classes()

        # --- Block 1 - Hardware Target (Weight extraction layer for Amaranth) ---
        # BatchNorm2d normalizes activations after each conv layer, reducing
        # internal covariate shift and stabilizing gradient flow during training.
        # Doubling filters at each block compensates for spatial resolution loss
        # from MaxPool, preserving representational capacity across the network.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 2 - Deeper feature extraction ---
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 3 - Adaptation & Classification (AdaptiveAvgPool to force 4x4 output) ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        # Dropout(p=0.3) before the FC layer randomly deactivates 30% of neurons
        # during training, acting as a regularizer to reduce overfitting.
        # 64 channels * output (4 * 4) = 1024
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Applies two convolutional blocks (Conv → BatchNorm → ReLU → MaxPool),
        followed by adaptive average pooling, flattening, dropout regularization,
        and a fully connected classification head.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Prediction logits of shape (Batch, num_classes)
        """
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_hardware_target_weights(self, filter_index: int = 0) -> torch.Tensor:
        """
        Utility method for the Software -> Hardware bridge.

        Extracts weights from a specific filter in the first convolutional layer
        to prepare them for quantization towards Amaranth. This method is
        intentionally defined only on SimpleCNN — ResNet variants are not
        hardware targets and do not expose this interface.

        Args:
            filter_index (int): Index of the filter to extract (default is 0)

        Returns:
            torch.Tensor: Filter weights of shape (out_channels, Height, Width)
        """
        with torch.no_grad():
            weights = self.conv1.weight[filter_index].cpu().clone()
        return weights


def get_model(num_classes: int = 0) -> nn.Module:
    """
    Factory function returning the appropriate model for the configured dataset.

    Dispatches based on cfg.ml.dataset:
    - cifar10       → SimpleCNN (lightweight, hardware-bridge compatible)
    - tiny-imagenet → ResNet-18 (deeper architecture for 200-class task)
    - imagenet      → ResNet-50 with default ImageNet weights, FC replaced
                       if num_classes differs from 1000

    The hardware MAC testbench always targets SimpleCNN.conv1 regardless
    of the active training model — the HW/SW bridge is dataset-agnostic
    by design and must always be validated against SimpleCNN directly.

    Args:
        num_classes (int): Override number of output classes. 0 = auto-detect
                           from cfg.ml.dataset via get_num_classes().

    Returns:
        nn.Module: The instantiated model ready for .to(device).

    Raises:
        ValueError: If cfg.ml.dataset has no registered model.
    """
    n = num_classes or get_num_classes()

    if cfg.ml.dataset == "cifar10":
        return SimpleCNN(num_classes=n)

    elif cfg.ml.dataset == "tiny-imagenet":
        # ResNet-18 trained from scratch — no pretrained weights for Tiny-ImageNet
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
        return model

    elif cfg.ml.dataset == "imagenet":
        # ResNet-50 with default ImageNet weights for fine-tuning or full training
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        if n != 1000:
            model.fc = nn.Linear(model.fc.in_features, n)
        return model

    else:
        raise ValueError(
            f"No model registered for dataset '{cfg.ml.dataset}'. "
            "Choose from: 'cifar10', 'tiny-imagenet', 'imagenet'."
        )
