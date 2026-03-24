# ===========================================================
#  File    : dataset.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-25
# ===========================================================

import logging
import shutil
import zipfile
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import cfg

logger = logging.getLogger(__name__)

# ── Normalization statistics per dataset ──────────────────
# These datasets are well known, so finding official normalizations is easy.
# For custom datasets, compute per-channel statistics with a loop over the DataLoader:
# For mean → sum(x) / n_pixels  (per channel: R, G, B)
# For std  → sqrt(sum((x - mean)²) / n_pixels)  (per channel: R, G, B)
STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "tiny-imagenet": {
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
    },
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}

# ── Number of classes per dataset ─────────────────────────
NUM_CLASSES = {
    "cifar10": 10,
    "tiny-imagenet": 200,
    "imagenet": 1000,
}

# ── Input crop sizes ──────────────────────────────────────
CROP_SIZE = {
    "cifar10": 32,
    "tiny-imagenet": 64,
    "imagenet": 224,
}


def _get_cifar10(data_dir: Path) -> tuple:
    """Downloads and returns CIFAR-10 train/test datasets."""
    stats = STATS["cifar10"]
    size = CROP_SIZE["cifar10"]

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )

    train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return train, test


def _get_tiny_imagenet(data_dir: Path) -> tuple:
    """
    Downloads and prepares the Tiny-ImageNet dataset (200 classes, 64x64).

    Downloads the zip archive from Stanford's CS231n CDN if not already
    present, extracts it, and restructures the validation split into
    per-class subdirectories compatible with ImageFolder.
    """
    import urllib.request

    tiny_dir = data_dir / "tiny-imagenet-200"
    zip_path = data_dir / "tiny-imagenet-200.zip"
    url = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if not tiny_dir.exists():
        if not zip_path.exists():
            logger.info(f"Downloading Tiny-ImageNet from {url}...")
            urllib.request.urlretrieve(url, zip_path)
        logger.info("Extracting Tiny-ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
        logger.info("Tiny-ImageNet extracted.")

    # Restructure val split for ImageFolder compatibility
    val_dir = tiny_dir / "val"
    val_images_dir = val_dir / "images"
    val_annotations = val_dir / "val_annotations.txt"

    if val_images_dir.exists():
        if val_annotations.exists():
            with open(val_annotations) as f:
                expected_count = sum(1 for _ in f)
        else:
            expected_count = 0
        moved_count = sum(
            len(list(d.iterdir()))
            for d in val_dir.iterdir()
            if d.is_dir() and d.name != "images"
        )

        if moved_count < expected_count:
            logger.info("Restructuring Tiny-ImageNet val split...")
            with open(val_annotations) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_file = parts[0]
                    class_id = parts[1]
                    class_dir = val_dir / class_id
                    class_dir.mkdir(exist_ok=True)
                    src = val_images_dir / img_file
                    dst = class_dir / img_file
                    if src.exists():
                        shutil.move(str(src), str(dst))
            shutil.rmtree(str(val_images_dir))
            val_annotations.unlink()
            logger.info("Val split restructured.")
        else:
            logger.info("Val split already restructured — skipping.")

    stats = STATS["tiny-imagenet"]
    size = CROP_SIZE["tiny-imagenet"]

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )

    train = datasets.ImageFolder(
        root=str(tiny_dir / "train"), transform=transform_train
    )
    test = datasets.ImageFolder(root=str(val_dir), transform=transform_test)
    return train, test


def _get_imagenet(data_dir: Path) -> tuple:
    """
    Loads ImageNet from a local directory (auto-download not supported).

    ImageNet requires manual registration at https://image-net.org and
    cannot be downloaded automatically. The dataset must be pre-organized
    as follows before use:

        data_dir/imagenet/
            train/
                n01440764/  (synset folders)
                ...
            val/
                n01440764/
                ...

    Args:
        data_dir (Path): Root data directory containing the imagenet/ subfolder.

    Raises:
        FileNotFoundError: If the imagenet/train or imagenet/val directories
                           are not found at the expected path.
    """
    imagenet_dir = data_dir / "imagenet"
    train_dir = imagenet_dir / "train"
    val_dir = imagenet_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet not found at {imagenet_dir}.\n"
            "ImageNet cannot be downloaded automatically.\n"
            "Please register at https://image-net.org and place the dataset as:\n"
            f"  {train_dir}/\n"
            f"  {val_dir}/"
        )

    stats = STATS["imagenet"]
    size = CROP_SIZE["imagenet"]

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )

    train = datasets.ImageFolder(root=str(train_dir), transform=transform_train)
    test = datasets.ImageFolder(root=str(val_dir), transform=transform_test)
    return train, test


def get_num_classes() -> int:
    """
    Returns the number of classes for the configured dataset.

    If cfg.ml.num_classes is explicitly set to a non-zero value, that value
    takes precedence. Otherwise the standard class count for the dataset is used.

    Returns:
        int: Number of output classes.
    """
    if cfg.ml.num_classes != 0:
        return cfg.ml.num_classes
    return NUM_CLASSES.get(cfg.ml.dataset, 10)


def get_dataloaders(
    data_dir: str | None = None, dataset_override: str | None = None
) -> tuple[DataLoader, DataLoader]:
    """
    Factory function that returns DataLoaders for the configured dataset.

    Dispatches to the appropriate dataset loader based on cfg.ml.dataset.
    Supported values: 'cifar10', 'tiny-imagenet', 'imagenet'.

    Args:
        data_dir (str, optional): Override for the data directory.
                                  Defaults to cfg.ml.data_dir.
        dataset_override (str, optional): Force a specific dataset regardless
                                          of cfg.ml.dataset. Used by the testbench
                                          to load CIFAR-10 without mutating global cfg.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, test_loader)

    Raises:
        ValueError: If cfg.ml.dataset is not a supported value.
        FileNotFoundError: If ImageNet is selected but not found locally.
    """
    root = Path(data_dir or cfg.ml.data_dir)
    root.mkdir(parents=True, exist_ok=True)

    dataset = dataset_override or cfg.ml.dataset
    logger.info(f"Loading dataset: {dataset} from {root}")

    if dataset == "cifar10":
        train, test = _get_cifar10(root)
    elif dataset == "tiny-imagenet":
        train, test = _get_tiny_imagenet(root)
    elif dataset == "imagenet":
        train, test = _get_imagenet(root)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: 'cifar10', 'tiny-imagenet', 'imagenet'."
        )

    train_loader = DataLoader(
        train,
        batch_size=cfg.ml.batch_size,
        shuffle=True,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=cfg.ml.batch_size,
        shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=True,
    )

    return train_loader, test_loader
