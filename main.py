import sys
import traceback

import torch

from src.config import cfg
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN


def test_pipeline():
    print("=== Démarrage du diagnostics du pipeline ML ===\n")

    # --- 0. Détection du matériel ---
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Matériel Pytorch détecté: {device.upper()}")

    # --- 1. Diagnostic du DataLoader ---
    print("\nÉtape 1 : Test du DataLoader et téléchargement...")
    try:
        train_loader, _ = get_dataloaders(data_dir="./data")
        images, labels = next(iter(train_loader))

        expected_shape = (cfg.ml.batch_size, 3, 32, 32)
        assert (
            images.shape == expected_shape
        ), f"Erreur de dimension. Attendu: {expected_shape}, Obtenu: {images.shape}"
        assert (
            images.dtype == torch.float32
        ), f"Erreur de type. Les images doivent être en float32, Obtenu: {images.dtype}"

        print("✅ DataLoader OK (Dimensions et types corrects).")
    except Exception as e:
        print("\n❌ CRASH à l'Étape 1 : Problème avec les données.")
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Diagnostic du Modèle (Forward Pass) ---
    print("\nÉtape 2 : Test du Modèle CNN...")
    try:
        model = SimpleCNN()
        outputs = model(images)

        expected_out_shape = (cfg.ml.batch_size, cfg.ml.num_classes)
        assert (
            outputs.shape == expected_out_shape
        ), f"Erreur de dimension en sortie du modèle. Attendu: {expected_out_shape}, Obtenu: {outputs.shape}"

        print("✅ Modèle OK (Le Forward Pass a réussi sans erreur de tenseur).")
    except Exception as e:
        print("\n❌ CRASH à l'Étape 2 : Problème de dimension dans les couches du CNN.")
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Diagnostic du Pont Hardware ---
    print("\nÉtape 3 : Test de l'extraction pour Amaranth...")
    try:
        hw_weights = model.get_hardware_target_weights(filter_index=0)

        assert (
            len(hw_weights.shape) == 3
        ), f"Les poids extraits doivent être en 3D (Canaux, H, W). Obtenu: {hw_weights.shape}"

        print(
            f"✅ Extraction OK. (Dimensions du filtre isolé : {list(hw_weights.shape)})"
        )
    except Exception as e:
        print("\n❌ CRASH à l'Étape 3 : Impossible d'extraire les poids ciblés.")
        traceback.print_exc()
        sys.exit(1)

    print(
        "\nSUCCÈS TOTAL : Tous les composants ML sont validés et prêts pour l'entraînement !"
    )


if __name__ == "__main__":
    test_pipeline()
