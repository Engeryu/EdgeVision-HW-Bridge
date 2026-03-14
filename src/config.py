# ===========================================================
#  File    : config.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

from pydantic import BaseModel, Field


class MLConfig(BaseModel):
    """Configuration pour la partie Machine Learning (PyTorch)."""

    batch_size: int = Field(
        default=64, description="Taille du batch de la data training input"
    )
    epoch: int = Field(default=5, description="Nombre de passages sur le dataset")
    learning_rate: float = Field(
        default=1e-3, description="Taux d'apprentissage de l'optimiseur"
    )
    num_classes: int = Field(
        default=10, description="10 pour CIFAR-10, 200 pour Tiny-Imagenet"
    )


class HardwareConfig(BaseModel):
    """Configuration pour la partie Microélectronique (Amaranth)."""

    bit_width: int = Field(
        default=8, description="Largeur des bus de données en bits (e.g: int8)"
    )


class ProjectConfig(BaseModel):
    """Configuration globale unifiant le ML et le Hardware."""

    ml: MLConfig = Field(default_factory=MLConfig)
    hw: HardwareConfig = Field(default_factory=HardwareConfig)


# instance globale e.g: from src.config import cfg
cfg = ProjectConfig()
