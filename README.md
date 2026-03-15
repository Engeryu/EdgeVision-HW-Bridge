# 🌉 ASIC-Project: HW/SW Co-Design for Deep Learning

This repository demonstrates a complete end-to-end flow from training a Convolutional Neural Network (CNN) in PyTorch to validating its exact cycle-accurate execution on a simulated hardware accelerator designed in Amaranth HDL.

This project bridges the gap between **Data Science / AI** and **Digital ASIC/FPGA Design**, proving that software mathematical models can be perfectly translated into integer-based hardware logic.

## 🚀 Features

- **Hardware-Aware ML Architecture:** A custom lightweight CNN designed in PyTorch (`src.ml.model.SimpleCNN`) optimized for hardware deployment.
- **Automated Training Pipeline:** Full training loop on CIFAR-10 with validation and model checkpointing (`src.ml.train`).
- **Custom Hardware MAC Unit:** A cycle-accurate Multiply-Accumulate (MAC) unit described in Python using Amaranth HDL (`src.hardware.mac`), capable of exporting to Verilog.
- **Exact Co-Simulation:** A robust testbench (`src.hardware.testbench`) that extracts weights and image patches, quantizes them to `int8`, simulates the hardware clock cycle by cycle, and mathematically asserts that the Amaranth hardware output perfectly matches the PyTorch software prediction.
- **Modern Python Stack:** Managed entirely with `uv` for blazing-fast dependency resolution and `Pydantic` for strict configuration validation.

## 📂 Project Structure

```text
.
├── pyproject.toml        # Project configuration and dependencies (uv)
├── src/
│   ├── config.py         # Global configuration (Pydantic models)
│   ├── ml/               # Machine Learning Domain (PyTorch)
│   │   ├── dataset.py    # CIFAR-10 data fetching and augmentation
│   │   ├── model.py      # CNN Architecture & Hardware Target extraction
│   │   └── train.py      # Training loop and evaluation
│   └── hardware/         # Microelectronics Domain (Amaranth)
│       ├── mac.py        # Hardware MAC Unit design (Generates Verilog)
│       └── testbench.py  # Co-simulation and VCD waveform generation
```

## 🛠️ Getting Started

### 1. Prerequisites

Ensure you have [uv](https://docs.astral.sh/uv/) installed on your system.

This project is compatible with Python 3.14. However, due to PyTorch CUDA compatibility issues with specific older hardware (such as the GTX 1070 Max-Q), the environment is currently locked to Python 3.13.

### 2. Installation

Clone the repository and sync the dependencies:

```bash
git clone <your-repo-url>
cd asic-project
uv sync
```

_Note: If you want to use Python 3.14, you do not need to modify `pyproject.toml` (as it requires `python >= 3.13`). Just update the local version file:_

```bash
echo "3.14" > .python-version
uv sync
```

### 3. Download the Dataset

Before training or simulating, the CIFAR-10 dataset needs to be downloaded. This step will automatically create a `./data` directory in the root folder:

```bash
uv run python -c "from src.ml.dataset import get_dataloaders; get_dataloaders()"
```

_(Note: Running the training or testbench scripts will also trigger this download automatically if the folder is missing)._

### 4. Run the ML Training (Optional)

To train the CNN on CIFAR-10 and generate the model weights inside the `./checkpoints` directory:

```bash
uv run python -m src.ml.train
```

You will get an output of the training iteration with metrics, til seeing:

```bash
Model successfully saved at: ./checkpoints/cifar10.pth
The model is ready to be transferred to Hardware!
```

### 5. Generate the Hardware RTL (Verilog)

Before running the simulation, you can export the Amaranth hardware design into standard Verilog RTL. This proves the design is synthesizable for real FPGAs and ASICs:

```bash
uv run python -m src.hardware.mac
```

This will generate a `mac.v` file in the root directory.

### 6. Run the Hardware Co-Simulation (The Magic)

To extract the data, quantize it to `int8`, run the hardware simulation, and mathematically compare the Amaranth signals with the PyTorch tensors:

```bash
uv run python -m src.hardware.testbench
```

If successful, the console will output:

```text
Software Result (PyTorch): <value>
Hardware Result (Amaranth): <value>
Success: The Software/Hardware bridge is perfect!
Waveform file generated: 'mac_simulation.vcd'
```

### 7. Inspect the Hardware Waveforms

The simulation generates a standard `.vcd` file. You can open it with GTKWave to inspect the cycle-by-cycle electrical signals of the MAC unit:

```bash
# Install gtkwave from your package manager or from source ([https://gtkwave.github.io/gtkwave/install/unix_linux.html](https://gtkwave.github.io/gtkwave/install/unix_linux.html))
gtkwave mac_simulation.vcd
```

#### Optional tuto. How to use the GUI Software `GTKWave`

## 🧠 Technical Highlights

- **Quantization:** Bridging the gap between PyTorch's `float32` and the Hardware's `int8` data bus.
- **Synchronous Logic:** The Amaranth MAC unit uses a clock domain (`sync`) to accurately model pipeline latencies.
