"""The command line. One YAML plus one command reproduces any experiment."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import ExperimentConfig
from .estimate import estimate_sweep, format_estimate
from .hardware import make_backend
from .registry import probe
from .results import ResultStore
from .runner import run_sweep
from .sync import ResultSync

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
    for pass_config in experiment.passes:
        count = sum(1 for s in specs if s.mode == pass_config.mode)
        typer.echo(
            f"  pass {pass_config.mode:<12} batch {pass_config.batch_size:<3} "
            f"{pass_config.samples_per_task} samples x {pass_config.repeats} repeats -> {count} runs"
        )
    for method in experiment.methods:
        count = sum(1 for s in specs if s.method == method.name)
        typer.echo(f"  {method.name:<20} {method.kind:<12} {count} runs")

    dropped = experiment.dropped_cells()
    if dropped:
        typer.echo("\ndropped by method context limits (documented, not silent):")
        for name, context in dropped:
            typer.echo(f"  {name} at {context} tokens")


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
    sync: bool = typer.Option(False, help="Commit and push each record to a git remote as it lands."),
    remote: str = typer.Option("origin", help="Git remote to push results to."),
) -> None:
    """Execute a sweep, writing one record per run as it finishes."""
    experiment = ExperimentConfig.from_yaml(config)
    store = ResultStore(results)

    if dry_run:
        pending = store.pending(experiment.expand()) if resume else experiment.expand()
        for spec in pending[: limit or len(pending)]:
            typer.echo(spec.run_id)
        typer.echo(f"\n{len(pending)} runs pending")
        return

    def progress(index: int, total: int, spec) -> None:
        typer.echo(f"[{index}/{total}] {spec.slug}")

    syncer = None
    on_record = None
    if sync:
        syncer = ResultSync(store.root, remote=remote)
        ok, detail = syncer.available()
        if not ok:
            # Refuse up front rather than discovering at hour 12 that nothing
            # ever left the box.
            typer.echo(f"--sync requested but unusable: {detail}")
            raise typer.Exit(code=2)
        typer.echo(detail)
        on_record = lambda record: syncer.push(record.slug)  # noqa: E731

    with make_backend(experiment.runtime) as backend:
        records = run_sweep(
            experiment,
            store,
            backend=backend,
            resume=resume,
            limit=limit,
            progress=progress,
            on_record=on_record,
        )

    failed = [r for r in records if r.status != "ok"]
    typer.echo(f"\n{len(records)} runs, {len(failed)} failed -> {store.index_path}")
    if syncer:
        typer.echo(syncer.summary())
    for record in failed:
        first_line = (record.error or "").splitlines()[0] if record.error else "unknown"
        typer.echo(f"  FAILED {record.slug}: {first_line}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
