# ===========================================================
#  File    : dataset.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import cfg


def get_dataloaders(data_dir: str = "./data") -> tuple[DataLoader, DataLoader]:
    """
    Télécharge, prépare et retourne les DataLoaders pour le dataset CIFAR-10.

    Args:
        data_dir (str): Le dossier où seront stockées les images téléchargées.

    Returns:
        tuple: (train_loader, test_loader) prêts à être itérés.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Valeurs statistiques officielles de l'industrie pour la normalisation
    # Des moyenne et écart-type des canaux Rouge, Vert, Bleu ()
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # --- Pipeline d'entraînement (Data Augmentation) ---
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    # --- Pipeline de test (Inférence pure) ---
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    print(f"Vérification/Téléchargement de CIFAR-10 dans {data_dir}...")

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_test
    )

    # --- Création des itérateurs (DataLoaders) avec pin_memory=True ---
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
