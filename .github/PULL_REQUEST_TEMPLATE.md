## Description

<!-- What does this PR do? Be specific and concise. -->

## Type of Change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `refactor` — code restructuring without behavior change
- [ ] `docs` — documentation only
- [ ] `chore` — tooling, dependencies, configuration
- [ ] `perf` — performance improvement

## Scope

- [ ] `ml` — training pipeline, model, dataset
- [ ] `hw` — hardware units, testbenches, RTL
- [ ] `config` — configuration, presets
- [ ] `init` — initializer.sh, cleaner.sh
- [ ] `docs` — README, CHANGELOG, community files

## Checklist

- [ ] Targets `develop`, not `main`
- [ ] Co-simulation testbench passes: `uv run python -m src.hardware.testbenches.tb_mac`
- [ ] No bare `print()` in module code — `logging` used instead
- [ ] All new public functions and classes have docstrings
- [ ] `CHANGELOG.md` updated if this is a notable change
- [ ] Commit messages follow Conventional Commits format

## Testing

<!-- Describe how you tested this change. Include relevant output if applicable. -->

## Related Issues

<!-- Closes #N or References #N -->
