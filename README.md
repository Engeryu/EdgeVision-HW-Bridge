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
├── initializer.sh        # All-in-one executable script
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

### ⚡ Quick Start (Recommended)

The fastest way to run the full pipeline is via the provided shell script, which handles environment checks, dataset download, training, RTL generation, and co-simulation in one command:

```bash
chmod +x run.sh
./initializer.sh
```

To skip training and reuse an existing checkpoint:

```bash
./initializer.sh --skip-train
```

The script will guide you interactively if a checkpoint already exists (retrain or keep it).

> **Manual steps are documented below** for users who prefer to run each stage individually or integrate them into their own workflow.

### 1. Prerequisites

Ensure you have [uv](https://docs.astral.sh/uv/) installed on your system.

This project is compatible with Python 3.14. However, due to PyTorch CUDA compatibility issues with specific older hardware (such as the GTX 1070 Max-Q), the environment is currently locked to Python 3.13.

### 2. Installation

Clone the repository and sync the dependencies:

```bash
git clone git@github.com:Engeryu/EdgeVision-HW-Bridge.git
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

#### 🔍 Optional: How to use the GTKWave GUI

GTKWave can be intimidating at first glance. Follow these quick steps to visualize your MAC unit's clock cycles:

- Find the component: In the top-left pane (SST / Search Hierarchy), click on bench and then click on top.
- Select the signals: In the bottom-left pane, you will see all the pins of our MAC unit (clk, clear, pixel_in, weight_in, result_out).
- Append them: Select all these signals (using Shift+Click), then click the "Append" button at the bottom of that pane. They will appear in the main signal window.
- Adjust the view: Click the "Zoom Fit" button (an icon with four arrows pointing outwards in the top toolbar) to fit the entire simulation into your screen.
- Analyze: You can now click anywhere on the waveform graph to see the exact integer values of the pixels and weights at any given microsecond, and watch the accumulator (result_out) update on every rising edge of the clock (clk).

### 🔮 Future Work & Scaling

While this repository successfully demonstrates the core Software-to-Hardware bridge and cycle-accurate MAC operations, scaling this into a full-fledged Edge AI accelerator (ASIC/FPGA) would require the following architectural additions:

- On-Chip Memory (SRAM/BRAM): Integrating local memory blocks to store the quantized weights and input feature maps directly on the chip, reducing off-chip memory bottlenecks.
- Control Logic (FSM): Implementing a Finite State Machine to orchestrate the read/write addresses, controlling the loops over the image patches without needing Python to inject data cycle-by-cycle.
- Systolic Array / Spatial Architecture: Expanding the single MAC unit into a 2D array of MACs to process multiple pixels and filters in parallel, maximizing throughput.
- System Bus Integration: Wrapping the accelerator with an industry-standard bus interface (e.g., AXI4 or AXI-Stream) to allow a host processor (like an ARM Cortex or RISC-V) to offload ML tasks to our IP.

## 🧠 Technical Highlights

- **Quantization:** Bridging the gap between PyTorch's `float32` and the Hardware's `int8` data bus.
- **Synchronous Logic:** The Amaranth MAC unit uses a clock domain (`sync`) to accurately model pipeline latencies.
