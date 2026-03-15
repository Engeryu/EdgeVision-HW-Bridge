#!/usr/bin/env bash
# ===========================================================
#  File    : initializer.sh
#  Author  : engeryu
#  Created : 2026-03-15
#  Modified: 2026-03-15
# ===========================================================
#  Full pipeline runner for the EdgeVision HW/SW Co-Design project.
#  Executes each step sequentially with environment validation.
#
#  Usage:
#    ./run.sh           # Full pipeline (dataset + train + rtl + cosim)
#    ./run.sh --skip-train   # Skip training (use existing checkpoint)
# ===========================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helpers ───────────────────────────────────────────────
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}
section() {
    echo -e "\n${BLUE}══════════════════════════════════════${NC}"
    echo -e "${BLUE} $*${NC}"
    echo -e "${BLUE}══════════════════════════════════════${NC}"
}

# ── Argument parsing ──────────────────────────────────────
SKIP_TRAIN=false
for arg in "$@"; do
    case $arg in
    --skip-train) SKIP_TRAIN=true ;;
    *) error "Unknown argument: $arg. Usage: ./run.sh [--skip-train]" ;;
    esac
done

# ═══════════════════════════════════════════════════════════
# STEP 0 — Environment validation
# ═══════════════════════════════════════════════════════════
section "STEP 0 — Environment validation"

# Check uv
if ! command -v uv &>/dev/null; then
    error "'uv' is not installed. Install it via: curl -Lsf https://astral.sh/uv/install.sh | sh"
fi
success "uv found: $(uv --version)"

# Check pyproject.toml (we're in the right directory)
if [[ ! -f "pyproject.toml" ]]; then
    error "pyproject.toml not found. Please run this script from the project root."
fi
success "Project root confirmed."

# Sync dependencies (idempotent — safe to always run)
info "Syncing dependencies with uv..."
uv sync --quiet
success "Dependencies up to date."

# ═══════════════════════════════════════════════════════════
# STEP 1 — Dataset
# ═══════════════════════════════════════════════════════════
section "STEP 1 — CIFAR-10 Dataset"

if [[ -d "./data/cifar-10-batches-py" ]]; then
    warn "Dataset already present in ./data — skipping download."
else
    info "Downloading CIFAR-10..."
    uv run python -c "from src.ml.dataset import get_dataloaders; get_dataloaders()"
    success "Dataset ready."
fi

# ═══════════════════════════════════════════════════════════
# STEP 2 — ML Training
# ═══════════════════════════════════════════════════════════
section "STEP 2 — ML Training (PyTorch)"

if [[ "$SKIP_TRAIN" == true ]]; then
    if [[ ! -f "./checkpoints/cifar10.pth" ]]; then
        error "--skip-train requested but no checkpoint found at ./checkpoints/cifar10.pth"
    fi
    warn "--skip-train flag set, skipping training. Using existing checkpoint."
else
    if [[ -f "./checkpoints/cifar10.pth" ]]; then
        warn "Checkpoint already exists at ./checkpoints/cifar10.pth."
        read -r -p "        Retrain from scratch? [y/N] " answer
        if [[ "${answer,,}" != "y" ]]; then
            info "Keeping existing checkpoint."
        else
            info "Starting training..."
            uv run python -m src.ml.train
            success "Training complete. Checkpoint saved."
        fi
    else
        info "No checkpoint found. Starting training..."
        uv run python -m src.ml.train
        success "Training complete. Checkpoint saved."
    fi
fi

# ═══════════════════════════════════════════════════════════
# STEP 3 — RTL Generation (Verilog)
# ═══════════════════════════════════════════════════════════
section "STEP 3 — RTL Generation (Amaranth → Verilog)"

if [[ -f "./mac.v" ]]; then
    warn "mac.v already exists — regenerating."
fi

uv run python -m src.hardware.mac
success "Verilog file 'mac.v' generated."

# ═══════════════════════════════════════════════════════════
# STEP 4 — HW/SW Co-Simulation
# ═══════════════════════════════════════════════════════════
section "STEP 4 — HW/SW Co-Simulation (Amaranth Testbench)"

uv run python -m src.hardware.testbench
success "Co-simulation passed. Waveform saved as 'mac_simulation.vcd'."

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
section "Pipeline Complete ✓"
echo -e "  ${GREEN}✔${NC} Dataset       ./data/"
echo -e "  ${GREEN}✔${NC} Checkpoint    ./checkpoints/cifar10.pth"
echo -e "  ${GREEN}✔${NC} RTL           ./mac.v"
echo -e "  ${GREEN}✔${NC} Waveform      ./mac_simulation.vcd"
echo ""
info "Inspect waveforms with: gtkwave mac_simulation.vcd"
