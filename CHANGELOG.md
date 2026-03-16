# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project does not use semantic versioning — changes are tracked by milestone.

---

## [Unreleased]

### Planned

- FSM control unit for MAC orchestration
- SRAM/BRAM on-chip memory module
- Systolic array (2D MAC grid)
- AXI4 / AXI-Stream bus wrapper

---

## [2026-03-16] — Multi-dataset, model factory, training hardening

### Added

- Multi-dataset support: CIFAR-10, Tiny-ImageNet (auto-download), ImageNet (manual)
- Model factory `get_model()` dispatching SimpleCNN / ResNet-18 / ResNet-50 per dataset
- Dataset-aware hyperparameter presets (`DATASET_PRESETS` in `config.py`)
- Configurable optimizer: Adam and AdamW with weight decay
- Hardware package restructuring: `units/`, `bus/`, `testbenches/`
- `cleaner.sh` for artifact cleanup with `--force` flag
- `data_purge.py` for GPU VRAM and system RAM release
- `initializer.sh` multi-package-manager support (uv / poetry / conda / pip)
- `--dataset` and `--manager` flags in `initializer.sh`
- Community files: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`
- GitHub PR template and issue templates

### Changed

- `SimpleCNN` scaled from 16/32 to 32/64 filters (+5% val acc on CIFAR-10)
- `Trainer` class refactored with `Metrics` dataclass (from Jérémy Alcime's PR #4)
- Checkpoint now stores `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_acc`
- Checkpoint filename is now dataset-dependent (`cifar10.pth`, `tiny_imagenet.pth`)
- All `print()` replaced by `logging` module in `train.py`, `dataset.py`, `testbench.py`
- `train.py` now uses `apply_dataset_preset()` for automatic hyperparameter coordination

### Fixed

- Testbench always uses CIFAR-10 for pixel extraction regardless of active dataset
- `avg_loss()` now derives window from `metrics.total` instead of `batch_idx`
- Duplicate `metrics.loss` accumulation removed from `train_one_epoch()`
- Missing f-strings in logger calls corrected
- AMP `autocast` and `GradScaler` wired correctly in training loop

---

## [2026-03-14] — Initial release

### Added

- `SimpleCNN` CNN architecture (PyTorch) with hardware-aware design
- CIFAR-10 training pipeline with CosineAnnealingLR and model checkpointing
- `MACUnit` cycle-accurate MAC hardware unit in Amaranth HDL
- Verilog RTL export from Amaranth (`mac.v`)
- HW/SW co-simulation testbench with int8 quantization and assertion
- VCD waveform generation for GTKWave inspection
- Pydantic-based unified configuration (`src/config.py`)
- `uv` project management with `pyproject.toml`
- `README.md` with full setup, usage, and GTKWave guide
