"""The command line. One YAML plus one command reproduces any experiment."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from .config import ExperimentConfig
from .estimate import estimate_sweep, format_estimate
from .hardware import make_backend
from .registry import probe
from .results import ResultStore
from .runner import plan_specs, run_sweep

app = typer.Typer(add_completion=False, help="Energy-aware KV cache compression benchmark.")

ConfigArg = typer.Argument(..., help="Path to an experiment YAML.")
ResultsOpt = typer.Option("results", help="Results directory.")


@app.command()
def doctor(config: Path | None = typer.Option(None, help="Only probe the methods this config uses.")) -> None:
    """Report which methods the installed kvpress can actually run."""
    names = None
    if config is not None:
        names = [m.name for m in ExperimentConfig.from_yaml(config).methods]

    statuses = probe(names)
    width = max(len(s.name) for s in statuses)
    for status in statuses:
        mark = "ok  " if status.available else "DROP"
        typer.echo(f"{mark}  {status.name:<{width}}  {status.kind:<12}  {status.detail}")

    dropped = [s.name for s in statuses if not s.available]
    if dropped:
        typer.echo("")
        typer.echo(f"unavailable: {', '.join(dropped)} -- remove these from the config")
        raise typer.Exit(code=1)


@app.command()
def validate(config: Path = ConfigArg) -> None:
    """Check a config parses and expands, without touching a GPU."""
    experiment = ExperimentConfig.from_yaml(config)
    specs = experiment.expand()
    typer.echo(f"{experiment.name}: {len(specs)} runs across {len(experiment.methods)} methods")
    for method in experiment.methods:
        count = sum(1 for s in specs if s.method == method.name)
        typer.echo(f"  {method.name:<20} {method.kind:<12} {count} runs")

    exclusions = experiment.exclusions()
    if exclusions:
        typer.echo("\nexcluded cells (these will not run, and belong in the write-up):")
        for name, cell, reason in exclusions:
            typer.echo(f"  {name} @ {cell}: {reason}")


@app.command()
def plan(
    config: Path = ConfigArg,
    results: Path = ResultsOpt,
    hourly_usd: float | None = typer.Option(None, help="GPU price per hour, to cost the sweep."),
) -> None:
    """Expand a config and estimate what the remaining runs will cost."""
    experiment = ExperimentConfig.from_yaml(config)
    store = ResultStore(results)
    typer.echo(format_estimate(estimate_sweep(experiment, store, hourly_usd)))


@app.command()
def run(
    config: Path = ConfigArg,
    results: Path = ResultsOpt,
    resume: bool = typer.Option(True, help="Skip cells that already completed successfully."),
    limit: int | None = typer.Option(None, help="Stop after this many runs."),
    dry_run: bool = typer.Option(False, help="List what would run, then exit."),
    shard: str = typer.Option(
        "0/1",
        help="This worker's slice, as i/n. One worker per GPU in a multi-GPU box.",
    ),
    device_index: int | None = typer.Option(
        None, help="Override the CUDA and NVML device index for this worker."
    ),
    sync_cmd: str | None = typer.Option(
        None,
        help="Shell command run after every record, to copy results off the box "
        "(e.g. a git commit and push). Failures are reported, never fatal.",
    ),
) -> None:
    """Execute a sweep, writing one record per run as it finishes."""
    experiment = ExperimentConfig.from_yaml(config)
    store = ResultStore(results)
    index, count = _parse_shard(shard)

    if device_index is not None:
        experiment.runtime.device_index = device_index
        if experiment.runtime.device.startswith("cuda"):
            experiment.runtime.device = f"cuda:{device_index}"

    if dry_run:
        pending = plan_specs(experiment, store, resume=resume, limit=limit, shard=(index, count))
        for spec in pending:
            typer.echo(spec.run_id)
        typer.echo(f"\n{len(pending)} runs pending for shard {index}/{count}")
        return

    def progress(position: int, total: int, spec) -> None:
        typer.echo(f"[{position}/{total}] {spec.slug}")

    with make_backend(experiment.runtime) as backend:
        records = run_sweep(
            experiment,
            store,
            backend=backend,
            resume=resume,
            limit=limit,
            progress=progress,
            shard=(index, count),
            after_record=_make_sync(sync_cmd),
        )

    failed = [r for r in records if r.status != "ok"]
    typer.echo(f"\n{len(records)} runs, {len(failed)} failed -> {store.index_path}")
    for record in failed:
        first_line = (record.error or "").splitlines()[0] if record.error else "unknown"
        typer.echo(f"  FAILED {record.slug}: {first_line}")
    if failed:
        raise typer.Exit(code=1)


def _parse_shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in value.split("/", 1))
    except ValueError as exc:
        raise typer.BadParameter(f"shard must look like i/n, got {value!r}") from exc
    if count < 1 or not 0 <= index < count:
        raise typer.BadParameter(f"shard {value} is out of range")
    return index, count


def _make_sync(command: str | None):
    """Run a copy-off-the-box command after each record.

    Deliberately never fatal: losing the network for a minute should cost you a
    sync, not twenty hours of sweep.
    """
    if not command:
        return None

    def sync(record) -> None:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            typer.echo(f"  sync failed ({result.returncode}): {result.stderr.strip()[:200]}")

    return sync


if __name__ == "__main__":
    app()
