# Contributing to EdgeVision HW/SW Co-Design

Thank you for your interest in contributing. This document outlines the process and expectations for contributions.

## Before You Start

- Check existing [issues](https://github.com/Engeryu/EdgeVision-HW-Bridge/issues) and [pull requests](https://github.com/Engeryu/EdgeVision-HW-Bridge/pulls) to avoid duplicating work.
- For significant changes, open an issue first to discuss the approach before writing code.
- Read the [README](./README.md) to understand the project architecture and the HW/SW bridge design.

## Development Setup

```bash
git clone git@github.com:Engeryu/EdgeVision-HW-Bridge.git
cd EdgeVision-HW-Bridge
uv sync
```

Alternatively, use `poetry`, `conda`, or `pip` — see the README for details.

## Branching Strategy

- `main` — stable, production-ready code only
- `develop` — integration branch, all PRs target this branch
- Feature branches — `feat/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`

**Always branch from `develop`, never from `main`.**

```bash
git checkout develop
git checkout -b feat/my-feature
```

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `perf`

**Scopes:** `ml`, `hw`, `init`, `config`, `docs`

Examples:

```
feat(hw): add FSM control unit for MAC orchestration
fix(ml): correct avg_loss window calculation
docs: update README with new module paths
```

## Pull Request Process

1. Target the `develop` branch — PRs to `main` will be rejected.
2. Fill out the PR template completely.
3. Ensure your changes do not break the co-simulation testbench:
   ```bash
   uv run python -m src.hardware.testbenches.tb_mac
   ```
4. If you modify `SimpleCNN` or any hardware unit, re-run the full pipeline:
   ```bash
   ./initializer.sh --skip-download
   ```
5. One logical change per PR. Do not bundle unrelated fixes.

## Code Style

- Python code must be formatted with `ruff` or `black`.
- All public functions and classes require docstrings.
- Hardware modules (`units/`) must include a comment justifying any signal bit-width choice.
- No bare `print()` in module code — use `logging`.

## Areas Open for Contribution

| Area                                  | Status         |
| ------------------------------------- | -------------- |
| Additional hardware units (FSM, SRAM) | Welcome        |
| AXI4 / AXI-Stream bus wrapper         | Welcome        |
| Systolic array implementation         | Welcome        |
| Additional dataset support            | Welcome        |
| Performance benchmarks                | Welcome        |
| Documentation improvements            | Welcome        |
| Bug fixes                             | Always welcome |

## Questions

Open a [GitHub Discussion](https://github.com/Engeryu/EdgeVision-HW-Bridge/discussions) for general questions, or an [Issue](https://github.com/Engeryu/EdgeVision-HW-Bridge/issues) for specific bugs or feature requests.
