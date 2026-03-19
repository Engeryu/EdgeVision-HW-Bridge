#!/usr/bin/env bash
# ===========================================================
#  File    : cleaner.sh
#  Author  : engeryu
#  Created : 2026-03-15
# ===========================================================
#  Removes all runtime-generated files and directories.
#
#  Usage:
#    ./cleaner.sh           # Interactive (asks confirmation)
#    ./cleaner.sh --force   # Non-interactive (no prompt)
# ===========================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
    --force)
        FORCE=true
        shift
        ;;
    *) error "Unknown argument: $1. Usage: ./cleaner.sh [--force]" ;;
    esac
done

# ── Targets ───────────────────────────────────────────────
# Directories
DIRS=(
    "./data"
    "./checkpoints"
    "./.venv"
    "./__pycache__"
    "./src/__pycache__"
    "./src/ml/__pycache__"
    "./src/hardware/units/__pycache__"
    "./src/hardware/testbenches/__pycache__"
    "./.pytest_cache"
)

# Files
FILES=(
    "./mac.v"
    "./mac_simulation.vcd"
)

# ── Preview ───────────────────────────────────────────────
section "Targets to clean"

echo -e "\n  ${YELLOW}Directories:${NC}"
for d in "${DIRS[@]}"; do
    [[ -d "$d" ]] && echo -e "    ${RED}✗${NC} $d" || echo -e "    ${BLUE}—${NC} $d (not found)"
done

echo -e "\n  ${YELLOW}Files:${NC}"
for f in "${FILES[@]}"; do
    [[ -f "$f" ]] && echo -e "    ${RED}✗${NC} $f" || echo -e "    ${BLUE}—${NC} $f (not found)"
done

echo ""

# ── Confirmation ──────────────────────────────────────────
if [[ "$FORCE" == false ]]; then
    read -r -p "  Proceed with cleanup? [y/N] " answer
    [[ "${answer,,}" != "y" ]] && {
        info "Aborted."
        exit 0
    }
fi

# ── Cleanup ───────────────────────────────────────────────
section "Cleaning"

for d in "${DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        rm -rf "$d"
        success "Removed directory: $d"
    fi
done

# Also remove nested __pycache__ recursively (catches any depth)
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null || true

for f in "${FILES[@]}"; do
    if [[ -f "$f" ]]; then
        rm -f "$f"
        success "Removed file: $f"
    fi
done

section "Cleanup Complete ✓"
info "Run ./initializer.sh to regenerate all artifacts."
