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

from src.config import cfg
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Executes a single training epoch over the dataset.

    This function iterates through the provided training DataLoader, performs
    the forward pass, calculates the loss, and executes the backward pass to
    update the model's weights. It also logs the running loss and accuracy
    metrics to the console every 100 batches.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader providing the training batches.
        criterion (nn.Module): The loss function used for optimization.
        optimizer (optim.Optimizer): The optimizer algorithm (e.g., Adam) to update weights.
        device (torch.device): The hardware device (CPU, CUDA, MPS) to perform computations on.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            accuracy = 100.0 * correct / total
            print(
                f"  Batch {batch_idx:03d}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
            )
            running_loss = 0.0


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluates the model on the validation or test dataset.

    This function runs the model in evaluation mode (disabling dropout, batch norm updates)
    and within a `torch.no_grad()` context block to save memory and skip gradient computations.
    It calculates the overall accuracy and average loss across the entire dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        test_loader (torch.utils.data.DataLoader): The DataLoader providing the validation batches.
        criterion (nn.Module): The loss function used to evaluate model performance.
        device (torch.device): The hardware device for computation.

    Returns:
        tuple[float, float]: A tuple containing (average_loss, accuracy_percentage).
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * test_correct / test_total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy


def train_model(save_dir: str = "./checkpoints"):
    """
    Orchestrates the complete training pipeline for the CNN model.

    This function handles the high-level flow:
    1. Sets up the execution environment and target device.
    2. Initializes the dataloaders, model, loss function, and optimizer.
    3. Manages the epoch loop, calling the training and evaluation functions sequentially.
    4. Saves the final trained model weights to the specified directory.

    Args:
        save_dir (str, optional): The directory path where the model weights ('cifar10.pth')
                                  will be saved. Defaults to "./checkpoints".
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Starting training on: {device.type.upper()}")

    train_loader, test_loader = get_dataloaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.ml.learning_rate)

    print(
        f"Configuration: {cfg.ml.epoch} epochs | Batch Size: {cfg.ml.batch_size} | LR: {cfg.ml.learning_rate}"
    )

    for epoch in range(1, cfg.ml.epoch + 1):
        print(f"\n--- Iteration {epoch}/{cfg.ml.epoch} ---")
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        print(
            f"End of Epoch {epoch} | Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%\n"
        )

    save_path = f"{save_dir}/cifar10.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved at: {save_path}")
    print("The model is ready to be transferred to Hardware!")


if __name__ == "__main__":
    train_model()
