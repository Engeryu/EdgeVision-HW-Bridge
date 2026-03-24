# ===========================================================
#  File    : testbenches/tb_mac.py
#  Author  : engeryu
#  Created : 2026-03-14
#  Modified: 2026-03-25
# ===========================================================

import logging
from pathlib import Path

import torch
from amaranth.sim import Simulator, SimulatorContext

from src.config import cfg
from src.hardware.units.mac import MACUnit
from src.ml.dataset import get_dataloaders
from src.ml.model import SimpleCNN

logger = logging.getLogger(__name__)


def quantize_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulates hardware quantization by scaling floating-point values to 8-bit integers.

    This function performs a symmetric quantization. It finds the maximum absolute
    value in the tensor and scales all elements so they fit within the [-127, 127]
    range. This strictly mirrors the precision loss expected when deploying ML
    models to edge integer-only hardware accelerators.

    Args:
        tensor (torch.Tensor): The input tensor containing float32 values.

    Returns:
        torch.Tensor: The quantized tensor cast to torch.int8.
    """
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor.to(torch.int8)
    quantized = torch.round((tensor / max_val) * 127.0)
    return quantized.to(torch.int8)


def get_quantized_test_data() -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Extracts and prepares hardware-targeted weights and image patches for co-simulation.

    This utility function performs three main steps:
    1. Loads the model (from checkpoint if available, otherwise random weights)
       and extracts the 27 weights (3x3x3) from the first convolutional filter.
    2. Loads the CIFAR-10 dataset and extracts a corresponding 27-pixel patch
       (3x3 spatial area across 3 RGB channels) from the first image.
    3. Quantizes both the weights and pixels to int8, and computes the exact
       mathematical dot product expected from the hardware.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]: A tuple containing:
            - The flattened, quantized weight tensor.
            - The flattened, quantized pixel tensor.
            - The expected theoretical scalar result of the MAC operation.
    """
    logger.info("1. Loading PyTorch model and extracting data...")
    # Always use SimpleCNN for the HW bridge regardless of the training dataset.
    # The hardware MAC targets conv1 of SimpleCNN exclusively — ResNet variants
    # are production models and do not expose get_hardware_target_weights().
    model = SimpleCNN()
    ckpt = Path("./checkpoints/cifar10.pth")
    if ckpt.exists():
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info("  -> Loaded trained weights from checkpoint.")
    else:
        logger.info("  -> No checkpoint found, using random weights.")
    model.eval()
    sw_weights = model.get_hardware_target_weights(filter_index=0).flatten()

    train_loader, _ = get_dataloaders(dataset_override="cifar10")

    images, _ = next(iter(train_loader))
    sw_pixels = images[0, :, 0:3, 0:3].flatten()

    logger.info("2. Quantization from float32 to int8...")
    hw_weights = quantize_to_int8(sw_weights)
    hw_pixels = quantize_to_int8(sw_pixels)

    expected_sw_result = int(torch.dot(hw_weights.float(), hw_pixels.float()).item())
    return hw_weights, hw_pixels, expected_sw_result


def run_hardware_software_cosimulation() -> None:
    """
    Executes the end-to-end hardware-software co-simulation pipeline.

    This function acts as the definitive testbench for the Amaranth MACUnit design.
    It orchestrates the flow by:
    1. Fetching the quantized software reference data.
    2. Instantiating the Amaranth hardware MACUnit and binding it to a Simulator.
    3. Configuring a 1 MHz virtual clock.
    4. Injecting the quantized PyTorch data cycle-by-cycle into the simulated chip.
    5. Asserting that the final hardware accumulator register matches the software prediction.
    6. Generating a '.vcd' (Value Change Dump) file for GTKWave waveform analysis.

    Raises:
        AssertionError: If the final hardware computation differs from the software prediction.
    """
    print("=== Starting AI / Hardware Co-Simulation ===\n")

    # Part 1 & 2: Software Extraction & Bridge
    hw_weights, hw_pixels, expected_sw_result = get_quantized_test_data()
    print(f"  -> Expected mathematical result: {expected_sw_result}")

    # Part 3: Hardware Simulation
    logger.info("3. Starting hardware simulation (Amaranth)...")
    mac = MACUnit(bit_width=cfg.hw.bit_width)
    sim = Simulator(mac)
    sim.add_clock(1e-6)

    # Note: Amaranth requires the testbench process to be an async closure
    async def testbench_process(ctx: SimulatorContext) -> None:
        """Asynchronous closure driving the cycle-accurate simulation."""
        # Reset the accumulator
        ctx.set(mac.clear, 1)
        await ctx.tick()
        ctx.set(mac.clear, 0)

        print("  -> Injecting data into the chip (Cycle by cycle)...")
        # Feed pixels and weights sequentially
        for pixel, weight in zip(hw_pixels, hw_weights, strict=True):
            ctx.set(mac.pixel_in, int(pixel.item()))
            ctx.set(mac.weight_in, int(weight.item()))
            await ctx.tick()

        # Retrieve the final computed result
        hw_result = ctx.get(mac.result_out)

        print(f"\nSoftware Result (PyTorch): {expected_sw_result}")
        print(f"Hardware Result (Amaranth): {hw_result}")
        assert hw_result == expected_sw_result, "❌ DISASTER: Hardware mismatch!"
        print("Success: The Software/Hardware bridge is perfect!")

    sim.add_testbench(testbench_process)

    # Run the simulation and dump signals for inspection
    with sim.write_vcd("mac_simulation.vcd"):
        sim.run()

    print("Waveform file generated: 'mac_simulation.vcd'")


def main() -> None:
    run_hardware_software_cosimulation()


if __name__ == "__main__":
    main()
