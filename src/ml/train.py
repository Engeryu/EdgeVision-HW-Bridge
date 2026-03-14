# ===========================================================
#  File    : train.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import cfg
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN


def train_model(save_dir: str = "./checkpoints"):
    """Entraîne le modèle CNN sur CIFAR-10 et sauvegarde les poids."""
    # --- 1. Préparation de l'environnement ---
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Démarrage de l'entraînement sur: {device.type.upper()}")

    # --- 2. Chargement des données et du modèle ---
    train_loader, test_loader = get_dataloaders()
    model = SimpleCNN().to(device)

    # --- 3. Définiton de la Loss Function et de l'optimiseur (ADAM) ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.ml.learning_rate)

    print(
        f"Configuration: {cfg.ml.epoch} itérations | Batch Size: {cfg.ml.batch_size} | LR: {cfg.ml.learning_rate}"
    )

    # --- 4. Boucle d'entraînement ---
    for epoch in range(1, cfg.ml.epoch + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n--- Itération {epoch}/{cfg.ml.epoch} ---")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # --- Statistiques en temps réel ---
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Rafraichissement d'affichage tous les 100 batchs
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = running_loss / 100
                accuracy = 100.0 * correct / total
                print(
                    f"  Batch {batch_idx:03d}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
                )

        # --- 5. Évaluation sur le set de Test (Validation) à la fin de l'itération ---
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

        test_acc = 100.0 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        print(
            f"Fin Itération {epoch} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%\n"
        )

    # --- 6. Sauvegarde des poids du modèle ---
    save_path = f"{save_dir}/cifar10.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModèle sauvegardé avec succès dans: {save_path}")
    print("Le modèle est prêt à Être transférée vers le Hardware !")


if __name__ == "__main__":
    train_model()
