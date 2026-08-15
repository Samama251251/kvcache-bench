# Vendored scorers

`longbench_metrics.py` and `ruler_metrics.py` are copied **verbatim** from
NVIDIA/kvpress, Apache-2.0:

- `evaluation/benchmarks/longbench/calculate_metrics.py`
- `evaluation/benchmarks/ruler/calculate_metrics.py`

They are vendored rather than imported because kvpress's `evaluation/` directory
ships only in the git repository, not in the published wheel, and is not an
importable package even when the repo is checked out.

Using kvpress's own scoring is deliberate: it is what the published kvpress
numbers were produced with, so our baseline should be comparable to theirs. That
only holds if these files stay byte-identical to upstream — do not reformat,
lint or "improve" them. Ruff is configured to skip this directory. Any local
adaptation belongs in `../scoring.py`.

Vendored from kvpress 0.5.4 (repository `main`, August 2026).
