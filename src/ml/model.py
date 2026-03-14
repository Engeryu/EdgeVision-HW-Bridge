# ===========================================================
#  File    : model.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

import torch
import torch.nn as nn

from src.config import cfg


class SimpleCNN(nn.Module):
    """
    Réseau de neurones convolutif léger pour classification d'images.
    Conçu pour être facilement interfaçable avec notre module matériel MAC.
    """

    def __init__(self, num_classes: int = cfg.ml.num_classes):
        super().__init__()

        # --- Bloc 1 - Hardware Target (couche d'extraction des poids pour Amaranth)---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bloc 2 - Extraction de caractéristiques plus profondes ---
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bloc 3 - Adaptation et Classification (AdaptiveAvgPool pour forcer la sortie en taille `4, 4`)---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        # 32 canaux * output (4 * 4) = 512
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du modèle.

        Args:
            x (torch.Tensor): Tensor d'images de forme (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Logits de prédiction de forme (Batch, num_classes)
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_hardware_target_weights(self, filter_index: int = 0) -> torch.Tensor:
        """
        Méthode utilitaire pour le point Software -> Hardware.
        Extrait les poids d'un filtre spécifique de la première couche de convolution
        pour les préparer à la quantification vers Amaranth.

        Args:
            filter_index (int): L'index du filtre à extratire (par défaut le 1er)

        Returns:
            torch.Tensor: Les poids du filtre de forme (Channels, Height, Width)
        """
        with torch.no_grad():
            weights = self.conv1.weight[filter_index].cpu().clone()
        return weights
