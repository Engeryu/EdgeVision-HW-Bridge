#!/usr/bin/env bash
# ===========================================================
#  File    : initializer.sh
#  Author  : engeryu
#  Created : 2026-03-15
#  Modified: 2026-03-15
# ===========================================================
#  Full pipeline runner for the EdgeVision HW/SW Co-Design project.
#  Executes each step sequentially with environment validation.
#  Auto-detects package manager: uv > poetry > conda > pip
#
#  Usage:
#    ./initializer.sh                     # Full pipeline (auto-detect manager)
#    ./initializer.sh --skip-train        # Skip training (use existing checkpoint)
#    ./initializer.sh --manager uv        # Force a specific package manager
#    ./initializer.sh --manager poetry
#    ./initializer.sh --manager conda
#    ./initializer.sh --manager pip
# ===========================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
FORCED_MANAGER=""

for arg in "$@"; do
    case $arg in
    --skip-train) SKIP_TRAIN=true ;;
    --manager) shift ;; # handled below via index
    --manager=*) FORCED_MANAGER="${arg#*=}" ;;
    uv | poetry | conda | pip)
        # Positional value after --manager
        [[ -n "$FORCED_MANAGER" ]] || FORCED_MANAGER="$arg"
        ;;
    *) error "Unknown argument: $arg. Usage: ./initializer.sh [--skip-train] [--manager uv|poetry|conda|pip]" ;;
    esac
done

# Re-parse --manager <value> (two-token form)
for i in "${!@}"; do
    if [[ "${!i}" == "--manager" ]]; then
        next=$((i + 1))
        FORCED_MANAGER="${!next:-}"
        break
    fi
done 2>/dev/null || true

# ═══════════════════════════════════════════════════════════
# STEP 0 — Environment validation & package manager setup
# ═══════════════════════════════════════════════════════════
section "STEP 0 — Environment validation"

# Check pyproject.toml (we're in the right directory)
if [[ ! -f "pyproject.toml" ]]; then
    error "pyproject.toml not found. Please run this script from the project root."
fi
success "Project root confirmed."

# ── Package manager detection ─────────────────────────────
detect_manager() {
    if command -v uv &>/dev/null; then
        echo "uv"
    elif command -v poetry &>/dev/null; then
        echo "poetry"
    elif command -v conda &>/dev/null; then
        echo "conda"
    elif command -v pip &>/dev/null; then
        echo "pip"
    else
        echo ""
    fi
}

MANAGER="${FORCED_MANAGER:-$(detect_manager)}"

[[ -z "$MANAGER" ]] && error "No supported package manager found (uv / poetry / conda / pip). Please install one."

# Validate forced manager is actually available
if [[ -n "$FORCED_MANAGER" ]] && ! command -v "$FORCED_MANAGER" &>/dev/null; then
    error "'$FORCED_MANAGER' was requested via --manager but is not installed."
fi

success "Package manager: ${MANAGER}"

# ── Dependency sync & PYTHON_RUN setup ───────────────────
case "$MANAGER" in

uv)
    info "Syncing dependencies with uv..."
    uv sync --quiet
    PYTHON_RUN="uv run python"
    success "uv: $(uv --version) — dependencies up to date."
    ;;

poetry)
    if [[ ! -f "pyproject.toml" ]]; then
        error "poetry requires pyproject.toml (already confirmed above — should not happen)."
    fi
    info "Installing dependencies with poetry..."
    poetry install --quiet
    PYTHON_RUN="poetry run python"
    success "poetry: $(poetry --version) — dependencies up to date."
    ;;

conda)
    CONDA_ENV_NAME="edgevision"
    if conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
        info "Conda env '${CONDA_ENV_NAME}' already exists — updating..."
        conda env update -n "$CONDA_ENV_NAME" -f environment.yml --prune --quiet
    else
        if [[ ! -f "environment.yml" ]]; then
            error "environment.yml not found. Cannot create conda environment."
        fi
        info "Creating conda env '${CONDA_ENV_NAME}'..."
        conda env create -f environment.yml --quiet
    fi
    PYTHON_RUN="conda run --no-capture-output -n ${CONDA_ENV_NAME} python"
    success "conda: $(conda --version) — env '${CONDA_ENV_NAME}' ready."
    ;;

pip)
    VENV_DIR=".venv"
    if [[ ! -f "requirements.txt" ]]; then
        error "requirements.txt not found. Cannot install with pip."
    fi
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating virtual environment in ${VENV_DIR}..."
        python -m venv "$VENV_DIR"
    else
        info "Virtual environment ${VENV_DIR} already exists."
    fi
    info "Installing dependencies with pip..."
    "${VENV_DIR}/bin/pip" install -r requirements.txt --quiet
    PYTHON_RUN="${VENV_DIR}/bin/python"
    success "pip: $("${VENV_DIR}/bin/pip" --version) — dependencies up to date."
    ;;
esac

# ═══════════════════════════════════════════════════════════
# STEP 1 — Dataset
# ═══════════════════════════════════════════════════════════
section "STEP 1 — CIFAR-10 Dataset"

if [[ -d "./data/cifar-10-batches-py" ]]; then
    warn "Dataset already present in ./data — skipping download."
else
    info "Downloading CIFAR-10..."
    $PYTHON_RUN -c "from src.ml.dataset import get_dataloaders; get_dataloaders()"
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
            $PYTHON_RUN -m src.ml.train
            success "Training complete. Checkpoint saved."
        fi
    else
        info "No checkpoint found. Starting training..."
        $PYTHON_RUN -m src.ml.train
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

$PYTHON_RUN -m src.hardware.mac
success "Verilog file 'mac.v' generated."

# ═══════════════════════════════════════════════════════════
# STEP 4 — HW/SW Co-Simulation
# ═══════════════════════════════════════════════════════════
section "STEP 4 — HW/SW Co-Simulation (Amaranth Testbench)"

$PYTHON_RUN -m src.hardware.testbench
success "Co-simulation passed. Waveform saved as 'mac_simulation.vcd'."

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
section "Pipeline Complete ✓"
echo -e "  ${GREEN}✔${NC} Package manager  ${MANAGER}"
echo -e "  ${GREEN}✔${NC} Dataset          ./data/"
echo -e "  ${GREEN}✔${NC} Checkpoint       ./checkpoints/cifar10.pth"
echo -e "  ${GREEN}✔${NC} RTL              ./mac.v"
echo -e "  ${GREEN}✔${NC} Waveform         ./mac_simulation.vcd"
echo ""
info "Inspect waveforms with: gtkwave mac_simulation.vcd"
