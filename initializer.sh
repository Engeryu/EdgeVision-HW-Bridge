#!/usr/bin/env bash
# ===========================================================
#  File    : initializer.sh
#  Author  : engeryu
#  Created : 2026-03-15
#  Modified: 2026-03-16
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
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
    --skip-train)
        SKIP_TRAIN=true
        shift
        ;;
    --dataset)
        [[ -z "${2:-}" ]] && error "--dataset requires a value (cifar10|tiny-imagenet|imagenet)"
        DATASET="$2"
        shift 2
        ;;
    --dataset=*)
        DATASET="${1#*=}"
        shift
        ;;
    --manager)
        [[ -z "${2:-}" ]] && error "--manager requires a value (uv|poetry|conda|pip)"
        FORCED_MANAGER="$2"
        shift 2
        ;;
    --manager=*)
        FORCED_MANAGER="${1#*=}"
        shift
        ;;
    *)
        error "Unknown argument: $1. Usage: ./initializer.sh [--skip-train] [--dataset cifar10|tiny-imagenet|imagenet] [--manager uv|poetry|conda|pip]"
        ;;
    esac
done

# Validate --dataset value if provided
if [[ -n "$DATASET" ]]; then
    case "$DATASET" in
    cifar10 | tiny-imagenet | imagenet) ;;
    *) error "Invalid dataset '$DATASET'. Choose from: cifar10, tiny-imagenet, imagenet" ;;
    esac
fi

# Validate --manager value if provided
if [[ -n "$FORCED_MANAGER" ]]; then
    case "$FORCED_MANAGER" in
    uv | poetry | conda | pip) ;;
    *) error "Invalid manager '$FORCED_MANAGER'. Choose from: uv, poetry, conda, pip" ;;
    esac
fi

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
    uv sync
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

# Resolve active dataset: CLI flag > cfg.ml.dataset
if [[ -z "$DATASET" ]]; then
    DATASET=$($PYTHON_RUN -c "from src.config import cfg; print(cfg.ml.dataset)")
fi

section "STEP 1 — Dataset (${DATASET})"

case "$DATASET" in

cifar10)
    if [[ -d "./data/cifar-10-batches-py" ]]; then
        warn "CIFAR-10 already present in ./data — skipping download."
    else
        info "Downloading CIFAR-10..."
        $PYTHON_RUN -c "from src.ml.dataset import get_dataloaders; get_dataloaders()"
        success "CIFAR-10 ready."
    fi
    ;;

tiny-imagenet)
    if [[ -d "./data/tiny-imagenet-200" ]]; then
        warn "Tiny-ImageNet already present in ./data — skipping download."
    else
        info "Downloading Tiny-ImageNet (~236 MB)..."
        $PYTHON_RUN -c "from src.ml.dataset import get_dataloaders; get_dataloaders()"
        success "Tiny-ImageNet ready."
    fi
    ;;

imagenet)
    if [[ -d "./data/imagenet/train" && -d "./data/imagenet/val" ]]; then
        warn "ImageNet already present in ./data/imagenet — skipping."
    else
        error "ImageNet cannot be downloaded automatically.\nPlease register at https://image-net.org and place the dataset at:\n  ./data/imagenet/train/\n  ./data/imagenet/val/"
    fi
    ;;
esac

# ═══════════════════════════════════════════════════════════
# STEP 2 — ML Training
# ═══════════════════════════════════════════════════════════
section "STEP 2 — ML Training (PyTorch)"

CKPT_NAME=$($PYTHON_RUN -c "from src.config import cfg; print(cfg.ml.dataset.replace('-','_'))")
CKPT_PATH="./checkpoints/${CKPT_NAME}.pth"

if [[ "$SKIP_TRAIN" == true ]]; then
    if [[ ! -f "./checkpoints/${CKPT_NAME}.pth" ]]; then
        error "--skip-train requested but no checkpoint found at ./checkpoints/${CKPT_NAME}.pth"
    fi
    warn "--skip-train flag set, skipping training. Using existing checkpoint."
else
    if [[ -f "./checkpoints/${CKPT_NAME}.pth" ]]; then
        warn "Checkpoint already exists at ./checkpoints/${CKPT_NAME}.pth."
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
echo -e "  ${GREEN}✔${NC} Dataset          ./data/ (${DATASET})"
echo -e "  ${GREEN}✔${NC} Checkpoint       ./checkpoints/${CKPT_NAME}.pth"
echo -e "  ${GREEN}✔${NC} RTL              ./mac.v"
echo -e "  ${GREEN}✔${NC} Waveform         ./mac_simulation.vcd"
echo ""
info "Inspect waveforms with: gtkwave mac_simulation.vcd"
