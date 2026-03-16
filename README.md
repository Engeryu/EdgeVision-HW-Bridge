# 🌉 EdgeVision HW/SW Co-Design for Deep Learning

This repository demonstrates a complete end-to-end flow from training a Convolutional Neural Network (CNN) in PyTorch to validating its exact cycle-accurate execution on a simulated hardware accelerator designed in Amaranth HDL.

This project bridges the gap between **Data Science / AI** and **Digital ASIC/FPGA Design**, proving that software mathematical models can be perfectly translated into integer-based hardware logic.

## 🚀 Features

- **Hardware-Aware ML Architecture:** A custom lightweight CNN designed in PyTorch (`src.ml.model.SimpleCNN`) optimized for hardware deployment.
- **Automated Training Pipeline:** Full training loop on CIFAR-10 with validation and model checkpointing (`src.ml.train`).
- **Custom Hardware MAC Unit:** A cycle-accurate Multiply-Accumulate (MAC) unit described in Python using Amaranth HDL (`src.hardware.mac`), capable of exporting to Verilog.
- **Exact Co-Simulation:** A robust testbench (`src.hardware.testbench`) that extracts weights and image patches, quantizes them to `int8`, simulates the hardware clock cycle by cycle, and mathematically asserts that the Amaranth hardware output perfectly matches the PyTorch software prediction.
- **Modern Python Stack:** Managed entirely with `uv` for blazing-fast dependency resolution and `Pydantic` for strict configuration validation.

## 🧠 Technical Highlights

- **Quantization:** Bridging the gap between PyTorch's `float32` and the Hardware's `int8` data bus.
- **Synchronous Logic:** The Amaranth MAC unit uses a clock domain (`sync`) to accurately model pipeline latencies.

## 📂 Project Structure

```text
.
├── pyproject.toml        # Project configuration and dependencies (uv/poetry/pip)
├── requirements.txt      # pip-compatible dependency export
├── environment.yml       # conda environment definition
├── initializer.sh        # All-in-one pipeline runner (uv/poetry/conda/pip)
├── cleaner.sh            # Removes all runtime-generated files and directories
├── data_purge.py         # Releases GPU VRAM and system RAM held by PyTorch
├── mac.v                 # Generated Verilog RTL (Amaranth export)
├── mac_simulation.vcd    # Hardware waveform dump (GTKWave)
└── src/
    ├── config.py         # Global configuration (Pydantic models)
    ├── ml/               # Machine Learning Domain (PyTorch)
    │   ├── dataset.py    # Multi-dataset factory (CIFAR-10, Tiny-ImageNet, ImageNet)
    │   ├── model.py      # CNN Architecture & Hardware Target extraction
    │   └── train.py      # Training loop and evaluation
    └── hardware/         # Microelectronics Domain (Amaranth)
        ├── mac.py        # Hardware MAC Unit design (Generates Verilog)
        └── testbench.py  # Co-simulation and VCD waveform generation
```

## 🛠️ Getting Started

### 1. Prerequisites

This project requires **Python ≥ 3.13**, as i use a GTX 1070 max-q (latest cuda not supported), however, the project is fully compatible with **Python 3.14**.
Ensure you have at least one of the following package managers installed:

| Tool                 | Install                                                         |
| -------------------- | --------------------------------------------------------------- |
| `uv` _(recommended)_ | [astral.sh/uv](https://docs.astral.sh/uv/)                      |
| `poetry`             | [python-poetry.org](https://python-poetry.org/docs/)            |
| `conda`              | [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) |
| `pip`                | Bundled with Python ≥ 3.13                                      |

> **Manual command substitution table** — replace `uv run python` in all steps below according to your manager:
>
> | Manager  | Replace `uv run python` with          |
> | -------- | ------------------------------------- |
> | `uv`     | `uv run python`                       |
> | `poetry` | `poetry run python`                   |
> | `conda`  | `conda run -n edgevision python`      |
> | `pip`    | `source .venv/bin/activate && python` |

### ⚡ Quick Start (Recommended)

The fastest way to run the full pipeline is via the provided shell script, which auto-detects your package manager and handles environment checks, dataset download, training, RTL generation, and co-simulation in one command:

```bash
chmod +x initializer.sh cleaner.sh
./initializer.sh
```

To skip training and reuse an existing checkpoint:

```bash
./initializer.sh --skip-train
```

To force a specific package manager:

```bash
./initializer.sh --manager pip
./initializer.sh --manager conda
./initializer.sh --manager poetry
```

The script will guide you interactively if a checkpoint already exists (retrain or keep it).

> **Manual steps are documented below** for users who prefer to run each stage individually or integrate them into their own workflow.

---

### 2. Installation

Clone the repository:

```bash
git clone git@github.com:Engeryu/EdgeVision-HW-Bridge.git
cd EdgeVision-HW-Bridge
```

Then install dependencies with your package manager:

```bash
# uv (recommended)
uv sync

# poetry
poetry install

# conda
conda env create -f environment.yml
conda activate edgevision

# pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **uv only** — to use Python 3.14 instead of the default 3.13:
>
> ```bash
> echo "3.14" > .python-version
> uv sync
> ```

### 3. Configure and Download the Dataset

The active dataset is configured via `cfg.ml.dataset` in `src/config.py`. Three datasets are supported:

| Dataset         | Classes | Resolution | Download            |
| --------------- | ------- | ---------- | ------------------- |
| `cifar10`       | 10      | 32×32      | Automatic           |
| `tiny-imagenet` | 200     | 64×64      | Automatic (~236 MB) |
| `imagenet`      | 1000    | 224×224    | Manual (see below)  |

To trigger the download manually:

```bash
uv run python -c "from src.ml.dataset import get_dataloaders; get_dataloaders()"
```

> Replace `uv run python` with your manager's equivalent (see substitution table above).

_(Note: Running the training or testbench scripts will also trigger the download automatically if the dataset is missing.)_

> **ImageNet** cannot be downloaded automatically. Register at [image-net.org](https://image-net.org) and place the dataset at `./data/imagenet/train/` and `./data/imagenet/val/`.

### 4. Run the ML Training (Optional)

To train the CNN on CIFAR-10 and generate the model weights inside the `./checkpoints` directory:

```bash
uv run python -m src.ml.train
```

> Replace `uv run python` with your manager's equivalent (see substitution table above).

You will get an output of the training iteration with metrics, until seeing:

```text
Checkpoint saved: checkpoints/<dataset>.pth (epoch=N, acc=XX.XX%)
The model is ready to be transferred to Hardware!
Best Test Acc: XX.XX%
```

The active model is selected automatically based on the configured dataset:

| Dataset         | Model       | Notes                       |
| --------------- | ----------- | --------------------------- |
| `cifar10`       | `SimpleCNN` | Hardware-bridge compatible  |
| `tiny-imagenet` | `ResNet-18` | Trained from scratch        |
| `imagenet`      | `ResNet-50` | Pretrained ImageNet weights |

Hyperparameters (optimizer, LR, scheduler) are automatically preset per dataset. Manual overrides in `src/config.py` always take precedence.

### 5. Generate the Hardware RTL (Verilog)

Before running the simulation, you can export the Amaranth hardware design into standard Verilog RTL. This proves the design is synthesizable for real FPGAs and ASICs:

```bash
uv run python -m src.hardware.mac
```

> Replace `uv run python` with your manager's equivalent (see substitution table above).

This will generate a `mac.v` file in the root directory.

### 6. Run the Hardware Co-Simulation (The Magic)

To extract the data, quantize it to `int8`, run the hardware simulation, and mathematically compare the Amaranth signals with the PyTorch tensors:

```bash
uv run python -m src.hardware.testbench
```

> Replace `uv run python` with your manager's equivalent (see substitution table above).

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
# Install GTKWave from your package manager or from source:
# https://gtkwave.github.io/gtkwave/install/unix_linux.html
gtkwave mac_simulation.vcd
```

#### 🔍 Optional: How to use the GTKWave GUI

GTKWave can be intimidating at first glance. Follow these quick steps to visualize your MAC unit's clock cycles:

1. **Find the component:** In the top-left pane (SST / Search Hierarchy), click on `bench` then `top`.
2. **Select the signals:** In the bottom-left pane, you will see all the pins of our MAC unit (`clk`, `clear`, `pixel_in`, `weight_in`, `result_out`).
3. **Append them:** Select all signals (Shift+Click), then click the **Append** button at the bottom of that pane.
4. **Adjust the view:** Click the **Zoom Fit** button to fit the entire simulation into your screen.
5. **Analyze:** Click anywhere on the waveform to see the exact integer values at any given microsecond, and watch `result_out` update on every rising edge of `clk`.

---

## 🧹 Maintenance

### Clean generated artifacts

To remove all runtime-generated files and directories (`data/`, `checkpoints/`, `.venv/`, `__pycache__/`, `mac.v`, `*.vcd`):

```bash
./cleaner.sh          # Interactive — previews targets and asks for confirmation
./cleaner.sh --force  # Non-interactive — skips prompt
```

### Purge GPU VRAM and system RAM

After training or co-simulation, PyTorch may hold onto GPU memory. To release it without restarting your Python process:

```bash
uv run python data_purge.py
```

> Replace `uv run python` with your manager's equivalent (see substitution table above).

This runs Python's garbage collector, empties the CUDA/ROCm cache, resets memory statistics, and attempts to return fragmented RAM to the OS (Linux only via `malloc_trim`).

---

## 🔮 Future Work & Scaling

While this repository successfully demonstrates the core Software-to-Hardware bridge and cycle-accurate MAC operations, scaling this into a full-fledged Edge AI accelerator (ASIC/FPGA) would require the following architectural additions:

- **On-Chip Memory (SRAM/BRAM):** Integrating local memory blocks to store the quantized weights and input feature maps directly on the chip, reducing off-chip memory bottlenecks.
- **Control Logic (FSM):** Implementing a Finite State Machine to orchestrate the read/write addresses, controlling the loops over the image patches without needing Python to inject data cycle-by-cycle.
- **Systolic Array / Spatial Architecture:** Expanding the single MAC unit into a 2D array of MACs to process multiple pixels and filters in parallel, maximizing throughput.
- **System Bus Integration:** Wrapping the accelerator with an industry-standard bus interface (e.g., AXI4 or AXI-Stream) to allow a host processor (like an ARM Cortex or RISC-V) to offload ML tasks to our IP.
