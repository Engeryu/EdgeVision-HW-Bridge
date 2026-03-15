# ===========================================================
#  File    : config.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

from pydantic import BaseModel, Field


class MLConfig(BaseModel):
    """Machine Learning configuration (PyTorch)."""

    batch_size: int = Field(default=64, description="Data Training Input batch size")
    epoch: int = Field(default=5, description="Number of iteration over dataset")
    learning_rate: float = Field(default=1e-3, description="Optimizer's Learning Rate")
    num_classes: int = Field(
        default=10, description="10 for CIFAR-10, 200 for Tiny-Imagenet"
    )


class HardwareConfig(BaseModel):
    """Microelectronic part configuration (Amaranth)."""

    bit_width: int = Field(default=8, description="Data bus width in bits (e.g: int8)")


class ProjectConfig(BaseModel):
    """Unified configuration for ML & Hardware."""

    ml: MLConfig = Field(default_factory=MLConfig)
    hw: HardwareConfig = Field(default_factory=HardwareConfig)


# Globale Instance - e.g: from src.config import cfg
cfg = ProjectConfig()
