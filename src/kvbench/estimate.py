"""What a sweep will cost, before renting anything.

The GPU for this project is not chosen yet, and the choice should follow from a
number rather than a guess. `kvbench plan` expands a config, subtracts what is
already done, and prints GPU-hours and dollars.

Estimates are labelled by their basis. With no records in the store it is a
declared-throughput guess and says so; once real runs exist it switches to their
observed wall-clock times. Never presented as more certain than it is.
"""

from __future__ import annotations

import math
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
    """Mean estimated seconds per cell, averaged over the cells given."""
    if not specs:
        return config.estimate.fixed_overhead_s
    return sum(_declared_seconds_for(config, spec) for spec in specs) / len(specs)


def _declared_seconds_for(config: ExperimentConfig, spec: RunSpec) -> float:
    """One cell's cost.

    Prefill is compute-bound and scales with total tokens, so batching does not
    help it. Decode is bound by reading the weights, so a batch of N costs about
    the same as a batch of one -- which is the entire reason quality passes are
    batched and performance passes are not.
    """
    e = config.estimate
    prompt = spec.context_length or spec.max_context_tokens or e.assumed_prompt_tokens
    batches = math.ceil(spec.samples / max(1, spec.batch_size))

    prefill_s = spec.samples * prompt / e.prefill_tokens_per_s
    decode_s = batches * spec.max_new_tokens / e.decode_tokens_per_s
    return prefill_s + decode_s + e.fixed_overhead_s


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
