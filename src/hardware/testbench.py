# ===========================================================
#  File    : testbench.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-14
# ===========================================================

import torch
from amaranth.sim import Simulator

from src.config import cfg
from src.hardware.mac import MACUnit
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN


def quantize_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Simule une quantification: Ramène les valeurs entre -127 et 127."""
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor.to(torch.int8)
    quantized = torch.round((tensor / max_val) * 127.0)
    return quantized.to(torch.int8)


def run_hardware_software_cosimulation():
    print("=== Démarrage de la Co-Simulation IA / Hardware ===\n")

    # --- 1. Partie Software (Préparation des données) ---
    print("1. Chargement du modèle PyTorch et extraction...")
    model = SimpleCNN()

    sw_weights = model.get_hardware_target_weights(filter_index=0).flatten()

    train_loader, _ = get_dataloaders()
    images, _ = next(iter(train_loader))

    sw_pixels = images[0, :, 0:3, 0:3].flatten()

    # --- 2. Le pont (Quantification) ---
    print("2. Quantification de float32 vers int8...")
    hw_weights = quantize_to_int8(sw_weights)
    hw_pixels = quantize_to_int8(sw_pixels)

    expected_sw_result = int(torch.dot(hw_weights.float(), hw_pixels.float()).item())
    print(f"  -> Résultat mathématique attendu: {expected_sw_result}")

    # --- 3. Partie Hardware (Simulation) ---
    print("3. Démarrage de la simulation matérielle (Amaranth)...")
    mac = MACUnit(bit_width=cfg.hw.bit_width)
    sim = Simulator(mac)

    sim.add_clock(1e-6)

    async def testbench_process(ctx):
        ctx.set(mac.clear, 1)
        await ctx.tick()
        ctx.set(mac.clear, 0)

        print("  -> Injection des données dans la puce (Cycle par cycle)...")

        for count in range(len(hw_pixels)):
            p_val = int(hw_pixels[count].item())
            w_val = int(hw_weights[count].item())

            ctx.set(mac.pixel_in, p_val)
            ctx.set(mac.weight_in, w_val)

            await ctx.tick()

        hw_result = ctx.get(mac.result_out)

        print(f"\nRésultat Software (PyTorch): {expected_sw_result}")
        print(f"Résultat Hardware (Amaranth): {hw_result}")

        assert (
            hw_result == expected_sw_result
        ), "❌ DÉSASTRE : Le matériel ne calcule pas comme le logiciel !"
        print("Succès: Le pont Software/Hardware est parfait !")

    sim.add_testbench(testbench_process)

    with sim.write_vcd("mac_simulation.vcd"):
        sim.run()

    print("Fichier de chronogramme généré: 'mac_simulation.vcd'")


if __name__ == "__main__":
    run_hardware_software_cosimulation()
