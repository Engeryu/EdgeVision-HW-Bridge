# ===========================================================
#  File    : dataset.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-15
# ===========================================================

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import cfg


def get_dataloaders(data_dir: str = "./data") -> tuple[DataLoader, DataLoader]:
    """
    Downloads, prepares, and returns the DataLoaders for the CIFAR-10 dataset.

    Args:
        data_dir (str): The directory where the downloaded images will be stored.

    Returns:
        tuple: (train_loader, test_loader) ready for iteration.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Industry standard statistical values for CIFAR-10 normalization
    # Mean and standard deviation for the Red, Green, and Blue channels
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # --- Training Pipeline (Data Augmentation) ---
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    # --- Testing Pipeline (Pure Inference) ---
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    print(f"Checking/Downloading CIFAR-10 into {data_dir}...")

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # --- Creating iterators (DataLoaders) with pin_memory=True ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.ml.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.ml.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader
