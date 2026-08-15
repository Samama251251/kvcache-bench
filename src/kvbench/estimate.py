"""What a sweep will cost, before renting anything.

The GPU for this project is not chosen yet, and the choice should follow from a
number rather than a guess. `kvbench plan` expands a config, subtracts what is
already done, and prints GPU-hours and dollars.

Estimates are labelled by their basis. With no records in the store it is a
declared-throughput guess and says so; once real runs exist it switches to their
observed wall-clock times. Never presented as more certain than it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import ExperimentConfig, RunSpec
from .results import ResultStore, RunRecord

MIN_RECORDS_TO_CALIBRATE = 3


@dataclass
class SweepEstimate:
    total_cells: int
    completed_cells: int
    pending_cells: int
    seconds_per_run: float
    total_seconds: float
    hourly_usd: float | None
    basis: str
    note: str

    @property
    def hours(self) -> float:
        return self.total_seconds / 3600.0

    @property
    def cost_usd(self) -> float | None:
        if self.hourly_usd is None:
            return None
        return self.hours * self.hourly_usd


def estimate_sweep(
    config: ExperimentConfig, store: ResultStore, hourly_usd: float | None = None
) -> SweepEstimate:
    specs = config.expand()
    pending = store.pending(specs)
    hourly = hourly_usd if hourly_usd is not None else config.estimate.hourly_usd

    measured = _measured_seconds_per_run(store, config.model.id)
    if measured is not None:
        seconds, basis = measured, "measured"
        note = "from observed wall-clock time of completed runs on this model"
    else:
        seconds = _declared_seconds_per_run(config, pending or specs)
        basis = "declared"
        note = (
            "from declared throughput assumptions, not measurement -- re-run "
            "'kvbench plan' after the smoke session for a real number"
        )

    return SweepEstimate(
        total_cells=len(specs),
        completed_cells=len(specs) - len(pending),
        pending_cells=len(pending),
        seconds_per_run=seconds,
        total_seconds=seconds * len(pending),
        hourly_usd=hourly,
        basis=basis,
        note=note,
    )


def _measured_seconds_per_run(store: ResultStore, model_id: str) -> float | None:
    durations = [
        d
        for record in store.all_records()
        if record.status == "ok" and record.spec.get("model_id") == model_id
        if (d := _record_duration_s(record)) is not None
    ]
    if len(durations) < MIN_RECORDS_TO_CALIBRATE:
        return None
    return sum(durations) / len(durations)


def _record_duration_s(record: RunRecord) -> float | None:
    try:
        start = datetime.fromisoformat(record.started_at)
        end = datetime.fromisoformat(record.finished_at)
    except ValueError:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds > 0 else None


def _declared_seconds_per_run(config: ExperimentConfig, specs: list[RunSpec]) -> float:
    e = config.estimate
    prompt_tokens = [spec.context_length or e.assumed_prompt_tokens for spec in specs] or [
        e.assumed_prompt_tokens
    ]
    mean_prompt = sum(prompt_tokens) / len(prompt_tokens)

    per_sample = mean_prompt / e.prefill_tokens_per_s + (
        config.generation.max_new_tokens / e.decode_tokens_per_s
    )
    return per_sample * config.generation.samples_per_task + e.fixed_overhead_s


def format_estimate(estimate: SweepEstimate) -> str:
    lines = [
        f"cells:      {estimate.total_cells} total, "
        f"{estimate.completed_cells} done, {estimate.pending_cells} pending",
        f"per run:    {estimate.seconds_per_run / 60:.1f} min ({estimate.basis})",
        f"remaining:  {estimate.hours:.1f} GPU-hours",
    ]
    if estimate.cost_usd is not None:
        lines.append(f"cost:       ~${estimate.cost_usd:.2f} at ${estimate.hourly_usd:.2f}/hr")
    else:
        lines.append("cost:       set estimate.hourly_usd (or pass --hourly-usd) to price it")
    lines.append(f"basis:      {estimate.note}")
    return "\n".join(lines)
