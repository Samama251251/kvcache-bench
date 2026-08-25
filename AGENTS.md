# Repository Guidelines

## Project Structure & Module Organization

The installable Python package lives in `src/kvbench/`. Keep experiment orchestration in `runner.py`, configuration expansion in `config.py`, result persistence in `results.py`, and implementations behind the existing `methods/`, `metrics/`, `hardware/`, and `benchmarks/` interfaces. Add tests under `tests/`, mirroring the module name (for example, `src/kvbench/sync.py` is covered by `tests/test_sync.py`). YAML experiment definitions belong in `configs/`. Per-run JSON files in `results/runs/` are the source of truth; `results/index.csv` is derived. Do not reformat or casually edit `src/kvbench/benchmarks/vendored/`, which must remain identical to upstream.

## Build, Test, and Development Commands

- `uv sync`: install the package and development dependencies for local CPU/fake-hardware work.
- `uv sync --extra gpu`: add NVML support on an NVIDIA machine.
- `uv run kvbench validate configs/smoke.yaml`: validate and expand a sweep without running it.
- `uv run kvbench run configs/local_tiny.yaml`: exercise the local tiny-model workflow.
- `uv run pytest`: run the full test suite.
- `uv run ruff check src/ tests/`: run lint and import checks.
- `uv run ruff format --check src/ tests/`: verify formatting; omit `--check` to format edited files.
- `uv build`: build wheel and source distributions.

## Coding Style & Naming Conventions

Target Python 3.11–3.12, use four-space indentation, type hints, and focused modules. Ruff enforces a 110-character line length plus `E`, `F`, `I`, `UP`, and `B` rules. Use `snake_case` for modules, functions, fixtures, and config keys; `PascalCase` for classes; and descriptive test names beginning with `test_`. Keep metric collection method-agnostic and preserve the one-YAML/one-command experiment model.

## Testing Guidelines

Pytest discovers `tests/test_*.py`. Add regression tests for config validation, stable run IDs, atomic/resumable results, and method isolation when changing those areas. Use `FakeBackend` for local hardware tests. The end-to-end test may download a tiny Hugging Face model once and skips when offline; fake-backend energy values are never scientific measurements.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Add GPU-hour estimator and CLI`. Keep each commit to one coherent change. Pull requests should explain the purpose, affected configs or result schema, commands run, and any measurement implications. Link relevant issues and attach plots when reported outputs change. Never combine latency or energy results from different GPU models.

## Security & Configuration Tips

Keep Hugging Face tokens and machine credentials in environment variables, never YAML or result records. Preserve raw results, but do not commit caches, virtual environments, or model downloads.
