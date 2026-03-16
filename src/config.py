# ===========================================================
#  File    : config.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-16
# ===========================================================

from pydantic import BaseModel, Field


class MLConfig(BaseModel):
    """Machine Learning configuration (PyTorch)."""

    batch_size: int = Field(default=64, description="Data Training Input batch size.")
    epoch: int = Field(default=5, description="Number of iteration over dataset.")
    learning_rate: float = Field(default=1e-3, description="Optimizer's Learning Rate.")
    num_classes: int = Field(
        default=10, description="10 for CIFAR-10, 200 for Tiny-Imagenet."
    )
    scheduler: str = Field(
        default="plateau",
        description="LR Scheduler type: 'cosine' (CosineAnnealingLR) or 'plateau' (ReduceLROnPlateau).",
    )
    early_stopping_patience: int = Field(
        default=3,
        description="Epochs without improvement before stopping. Set to 0 to disable.",
    )
    seed: int = Field(
        default=42,
        description="Global random seed for reproducibility across torch, numpy and python random.",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable AMP (Automatic Mixed Precision) FP16 training via torch.amp.",
    )
    compile_model: bool = Field(
        default=False,
        description="Enable torch.compile() for kernel-level optimizations (PyTorch 2.0+). Adds ~30s warmup.",
    )


class HardwareConfig(BaseModel):
    """Microelectronic part configuration (Amaranth)."""

    bit_width: int = Field(default=8, description="Data bus width in bits (e.g: int8).")


class ProjectConfig(BaseModel):
    """Unified configuration for ML & Hardware."""

    ml: MLConfig = Field(default_factory=MLConfig)
    hw: HardwareConfig = Field(default_factory=HardwareConfig)


# Globale Instance - e.g: from src.config import cfg
cfg = ProjectConfig()
