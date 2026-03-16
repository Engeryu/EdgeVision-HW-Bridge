# ===========================================================
#  File    : model.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-16
# ===========================================================

import torch
import torch.nn as nn

from src.config import cfg


class SimpleCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network for image classification.
    Designed to easily interface with our hardware MAC module.
    """

    def __init__(self, num_classes: int = cfg.ml.num_classes):
        super().__init__()

        # --- Block 1 - Hardware Target (Weight extraction layer for Amaranth) ---
        # BatchNorm2d normalizes activations after each conv layer, reducing
        # internal covariate shift and stabilizing gradient flow during training.
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
        to prepare them for quantization towards Amaranth.

        Args:
            filter_index (int): Index of the filter to extract (default is 0)

        Returns:
            torch.Tensor: Filter weights of shape (Channels, Height, Width)
        """
        with torch.no_grad():
            weights = self.conv1.weight[filter_index].cpu().clone()
        return weights
