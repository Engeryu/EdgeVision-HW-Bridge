# ===========================================================
#  File    : train.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-16
# ===========================================================

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.config import cfg
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Sets the global random seed for full reproducibility.

    Applies the seed to Python's random module, NumPy, PyTorch CPU and
    CUDA backends. Also enables deterministic algorithm selection in
    cuDNN to eliminate non-determinism from GPU kernel choices.

    Args:
        seed (int): The seed value to apply globally.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}.")


@dataclass
class Metrics:
    """
    Accumulates training or evaluation statistics over a batch window.

    Tracks the running loss, the number of correct predictions, and the
    total number of samples seen. Provides convenience properties for
    accuracy and average loss computation, and a reset method to clear
    state between logging windows.
    """

    loss: float = 0.0
    correct: int = 0
    total: int = 0

    def reset(self) -> None:
        self.loss = 0.0
        self.correct = 0
        self.total = 0

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    def avg_loss(self) -> float:
        """Returns the mean loss per batch over the current accumulation window."""
        count = self.total / (cfg.ml.batch_size or 1)
        return self.loss / count if count else 0.0


class Trainer:
    """
    Orchestrates the complete training pipeline for the CNN model.

    Encapsulates the model, optimizer, scheduler, and data loaders as
    a single stateful object. Supports AMP mixed precision, configurable
    LR schedulers, patience-based early stopping, and fully resumable
    checkpoints. All behavior is driven by the global cfg instance.
    """

    def __init__(self, save_dir: str = "./checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        set_seed(cfg.ml.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.train_loader, self.test_loader = get_dataloaders()
        self.model = get_model().to(self.device)

        if cfg.ml.compile_model:
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.ml.learning_rate)

        if cfg.ml.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=1)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.ml.epoch)

        self.scaler = GradScaler(
            enabled=cfg.ml.mixed_precision and self.device.type == "cuda"
        )

    def train_one_epoch(self) -> None:
        """
        Executes a single training epoch over the full dataset.

        Iterates through the training DataLoader, performs the forward pass,
        computes the cross-entropy loss, and updates weights via backpropagation.
        Logs loss and accuracy every 100 batches and at the final batch of each epoch.
        Metrics are reset after each logging window to report per-window averages
        rather than cumulative values.
        """
        self.model.train()
        metrics = Metrics()

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=cfg.ml.mixed_precision):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics.loss += loss.item()
            _, predicted = outputs.max(1)
            metrics.total += labels.size(0)
            metrics.correct += predicted.eq(labels).sum().item()

            is_log_step = batch_idx % 100 == 0 and batch_idx > 0
            is_last_step = batch_idx == len(self.train_loader) - 1

            if is_log_step or is_last_step:
                logger.info(
                    f"  Batch {batch_idx + 1:03d}/{len(self.train_loader)} "
                    f"| Loss: {metrics.avg_loss():.4f} "
                    f"| Acc: {metrics.accuracy:.2f}%"
                )
                metrics.reset()

    def evaluate(self) -> tuple[float, float]:
        """
        Evaluates the model on the full test dataset.

        Runs the model in eval mode within a no_grad context to disable
        gradient computation and batch norm updates. Returns the average
        cross-entropy loss and the overall classification accuracy across
        all test batches.

        Returns:
            tuple[float, float]: (average_loss, accuracy_percentage)
        """
        self.model.eval()
        metrics = Metrics()

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                metrics.loss += loss.item()
                _, predicted = outputs.max(1)
                metrics.total += labels.size(0)
                metrics.correct += predicted.eq(labels).sum().item()

        return metrics.avg_loss(), metrics.accuracy

    def save(self, epoch: int, best_acc: float) -> None:
        """
        Saves a fully resumable checkpoint to the configured directory.

        Stores model weights, optimizer state, scheduler state, epoch index,
        and best validation accuracy. The optimizer and scheduler states allow
        exact training resumption from any checkpoint without loss of momentum
        or LR schedule position.

        Args:
            epoch (int): The epoch index at which the checkpoint is saved.
            best_acc (float): The best validation accuracy achieved so far.
        """
        save_path = self.save_dir / f"{cfg.ml.dataset.replace('-', '_')}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            },
            save_path,
        )
        logger.info(
            f"Checkpoint saved: {save_path} (epoch={epoch}, acc={best_acc:.2f}%)"
        )
        logger.info("The model is ready to be transferred to Hardware!")

    def run(self) -> None:
        """
        Orchestrates the full training pipeline.

        Manages the epoch loop, calling train_one_epoch and evaluate
        sequentially. Implements early stopping based on validation accuracy:
        if no improvement is observed for cfg.ml.early_stopping_patience
        consecutive epochs, training is halted. The best checkpoint is saved
        whenever a new accuracy peak is reached. Set early_stopping_patience
        to 0 in cfg to disable early stopping entirely.
        """
        logger.info(f"Starting training on: {self.device.type.upper()}")
        logger.info(
            f"Configuration: {cfg.ml.epoch} epochs | Dataset: {cfg.ml.dataset} "
            f"| Batch Size: {cfg.ml.batch_size} | LR: {cfg.ml.learning_rate} "
            f"| Scheduler: {cfg.ml.scheduler} | AMP: {cfg.ml.mixed_precision} "
            f"| Compile: {cfg.ml.compile_model}"
        )

        best_acc = 0.0
        patience_counter = 0
        patience = cfg.ml.early_stopping_patience

        for epoch in range(1, cfg.ml.epoch + 1):
            logger.info(f"\n--- Iteration {epoch}/{cfg.ml.epoch} ---")
            self.train_one_epoch()

            if isinstance(self.scheduler, ReduceLROnPlateau):
                val_loss, val_acc = self.evaluate()
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                val_loss, val_acc = self.evaluate()

            logger.info(
                f"End of Epoch {epoch} | Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%\n"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                self.save(epoch, best_acc)
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement for {patience_counter}/{patience} epoch(s)."
                )

            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        logger.info(f"Best Test Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    Trainer().run()
