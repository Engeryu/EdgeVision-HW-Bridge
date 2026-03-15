# ===========================================================
#  File    : train.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-15
# ===========================================================

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import cfg
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN


class Trainer:
    """Orchestrates the complete training pipeline for the CNN model."""

    def __init__(self, save_dir: str = "./checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.train_loader, self.test_loader = get_dataloaders()
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.ml.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.ml.epoch)

    def train_one_epoch(self) -> None:
        """Executes a single training epoch over the dataset."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = running_loss / 100
                accuracy = 100.0 * correct / total
                print(
                    f"  Batch {batch_idx:03d}/{len(self.train_loader)} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
                )
                running_loss = 0.0
                correct = 0
                total = 0

            running_loss += loss.item()

    def evaluate(self) -> tuple[float, float]:
        """Evaluates the model on the test dataset."""
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * test_correct / test_total
        avg_loss = test_loss / len(self.test_loader)
        return avg_loss, accuracy

    def save(self) -> None:
        """Saves the model weights to the configured directory."""
        save_path = self.save_dir / "cifar10.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model successfully saved at: {save_path}")
        print("The model is ready to be transferred to Hardware!")

    def run(self) -> None:
        """Runs the full training pipeline: epochs loop + evaluation + save."""
        print(f"Starting training on: {self.device.type.upper()}")
        print(
            f"Configuration: {cfg.ml.epoch} epochs | Batch Size: {cfg.ml.batch_size} | LR: {cfg.ml.learning_rate} (CosineAnnealing)"
        )

        for epoch in range(1, cfg.ml.epoch + 1):
            print(f"\n--- Iteration {epoch}/{cfg.ml.epoch} ---")
            self.train_one_epoch()
            self.scheduler.step()

            val_loss, val_acc = self.evaluate()
            print(
                f"End of Epoch {epoch} | Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%\n"
            )

        self.save()


if __name__ == "__main__":
    Trainer().run()
