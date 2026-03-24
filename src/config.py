# ===========================================================
#  File    : config.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-24
# ===========================================================

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MLConfig(BaseModel):
    """Machine Learning configuration (PyTorch)."""

    data_dir: str = Field(
        default="./data",
        description="Root directory for dataset storage.",
    )
    dataset: Literal["cifar10", "tiny-imagenet", "imagenet"] = Field(
        default="cifar10",
        description="Dataset to use: 'cifar10', 'tiny-imagenet', or 'imagenet'.",
    )
    num_classes: int = Field(
        default=0,
        description="Number of output classes. Set to 0 for auto-detection from dataset.",
    )
    scheduler: Literal["cosine", "plateau"] = Field(
        default="plateau",
        description=(
            "LR Scheduler type: 'cosine' (CosineAnnealingLR) or 'plateau' (ReduceLROnPlateau)."
        ),
    )
    seed: int = Field(
        default=42,
        description="Global random seed for reproducibility across torch, numpy and python random.",
    )
    batch_size: int = Field(default=64, description="Data Training Input batch size.")
    epoch: int = Field(default=5, description="Number of iteration over dataset.")
    optimizer: Literal["adam", "adamw"] = Field(
        default="adamw",
        description="Optimizer type: 'adam' or 'adamw'.",
    )
    learning_rate: float = Field(
        default=3e-4,
        description=(
            "Optimizer's initial learning rate. Recommended: 3e-4 for AdamW, 1e-3 for Adam."
        ),
    )
    weight_decay: float = Field(
        default=1e-4,
        description="L2 regularization factor for AdamW. Ignored when optimizer is 'adam'.",
    )
    early_stopping_patience: int = Field(
        default=3,
        description="Epochs without improvement before stopping. Set to 0 to disable.",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable AMP (Automatic Mixed Precision) FP16 training via torch.amp.",
    )
    compile_model: bool = Field(
        default=False,
        description="Enable torch.compile() for kernel-level optimizations (PyTorch 2.0+).",
    )


class HardwareConfig(BaseModel):
    """Microelectronic part configuration (Amaranth)."""

    bit_width: int = Field(default=8, description="Data bus width in bits (e.g: int8).")


class ProjectConfig(BaseModel):
    """Unified configuration for ML & Hardware.
    Mutable by design - required for apply_dataset_preset() in train.py.
    This is a global singleton: Safe for single-process use.
    However it would require dependency injection in multi-threaded contexts.
    """

    model_config = ConfigDict(frozen=False)

    ml: MLConfig = Field(default_factory=MLConfig)
    hw: HardwareConfig = Field(default_factory=HardwareConfig)


# Global Instance - e.g: from src.config import cfg
cfg = ProjectConfig()

# ── Dataset presets ───────────────────────────────────────
DATASET_PRESETS = {
    "cifar10": {
        "optimizer": "adamw",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    },
    "tiny-imagenet": {
        "optimizer": "adamw",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "plateau",
    },
    "imagenet": {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    },
}
