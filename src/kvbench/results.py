"""Run records: the actual deliverable.

Every run writes one self-describing JSON file the moment it finishes. A sweep
is therefore resumable, and a crash six hours in costs one run rather than the
session. Plots are regenerated from these files, never from re-running GPUs.

The schema is uniform and mandatory: every record carries all of quality,
memory, latency and energy. A field may be null where the device genuinely
cannot report it, but it can never be absent -- a partially-populated record
would quietly drop rows out of a cross-cutting comparison.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import RunSpec
from .metrics.energy import EnergyMetrics
from .metrics.latency import LatencyMetrics
from .metrics.memory import MemoryMetrics

SCHEMA_VERSION = 1

# The axes that make a record interpretable. Checked explicitly, because losing
# one of them silently is the failure mode that would waste a whole GPU session.
REQUIRED_AXES = ("quality", "memory", "latency", "energy")


class QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The benchmark's own automatic metric -- no LLM-as-a-judge anywhere.
    primary_metric: str
    primary_score: float | None
    scores: dict = Field(default_factory=dict)
    samples_scored: int


class MethodInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    press_class: str | None
    requested_compression_ratio: float | None
    # What the cache actually ended up at. Presses do not always hit the request
    # exactly (sink tokens, window minimums, per-layer budget rounding), and the
    # gap is itself a finding.
    effective_compression_ratio: float | None
    quant_bits: int | None
    params: dict = Field(default_factory=dict)


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str
    device_info: dict
    clocks_locked_mhz: int | None
    clock_lock_applied: bool
    torch_version: str | None
    transformers_version: str | None
    kvpress_version: str | None
    platform: str
    git_sha: str | None
    kvbench_version: str


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    slug: str
    spec: dict
    method: MethodInfo
    environment: Environment
    status: str
    error: str | None = None
    # Things that make this run less trustworthy without making it a failure:
    # too few power samples, a sampling rate far below what was asked for, NVML
    # read errors. A run that is quietly degraded is worse than one that failed.
    warnings: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str

    quality: QualityMetrics
    memory: MemoryMetrics
    latency: LatencyMetrics
    energy: EnergyMetrics

    def flat_row(self) -> dict:
        """One row for the index CSV: the headline numbers, already joined up."""
        return {
            "run_id": self.run_id,
            "experiment": self.spec.get("experiment"),
            "model_id": self.spec.get("model_id"),
            "method": self.method.name,
            "method_kind": self.method.kind,
            "retention": self.spec.get("retention"),
            "quant_bits": self.method.quant_bits,
            "effective_compression_ratio": self.method.effective_compression_ratio,
            "suite": self.spec.get("suite"),
            "task": self.spec.get("task"),
            "context_length": self.spec.get("context_length"),
            "repeat": self.spec.get("repeat"),
            "mode": self.spec.get("mode"),
            "batch_size": self.spec.get("batch_size"),
            "status": self.status,
            "warnings": "; ".join(self.warnings),
            "primary_metric": self.quality.primary_metric,
            "primary_score": self.quality.primary_score,
            "driver_peak_bytes": self.memory.driver_peak_bytes,
            "torch_peak_bytes": self.memory.torch_peak_bytes,
            "kv_cache_bytes": self.memory.kv_cache_bytes,
            "ttft_s": self.latency.ttft_s,
            "itl_mean_s": self.latency.itl_mean_s,
            "throughput_tokens_per_s": self.latency.throughput_tokens_per_s,
            "generated_tokens": self.latency.generated_tokens,
            "total_joules": self.energy.total_joules,
            "joules_per_generated_token": self.energy.joules_per_generated_token,
            "mean_watts": self.energy.mean_watts,
            "peak_watts": self.energy.peak_watts,
            "gpu": self.environment.device_info.get("name"),
            "hardware_backend": self.environment.device_info.get("backend"),
        }


INDEX_COLUMNS = [
    "run_id",
    "experiment",
    "model_id",
    "method",
    "method_kind",
    "retention",
    "quant_bits",
    "effective_compression_ratio",
    "suite",
    "task",
    "context_length",
    "repeat",
    "mode",
    "batch_size",
    "status",
    "warnings",
    "primary_metric",
    "primary_score",
    "driver_peak_bytes",
    "torch_peak_bytes",
    "kv_cache_bytes",
    "ttft_s",
    "itl_mean_s",
    "throughput_tokens_per_s",
    "generated_tokens",
    "total_joules",
    "joules_per_generated_token",
    "mean_watts",
    "peak_watts",
    "gpu",
    "hardware_backend",
]


class ResultStore:
    """Per-run JSON files plus a flat index, under one results directory."""

    def __init__(self, root: str | Path = "results") -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "index.csv"

    def path_for(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def save(self, record: RunRecord) -> Path:
        """Write the record atomically, then refresh the index."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(record.run_id)
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(record.model_dump(mode="json"), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        self.rebuild_index()
        return target

    def load(self, run_id: str) -> RunRecord:
        with open(self.path_for(run_id)) as fh:
            return RunRecord.model_validate(json.load(fh))

    def all_records(self) -> list[RunRecord]:
        if not self.runs_dir.exists():
            return []
        records = []
        for path in sorted(self.runs_dir.glob("*.json")):
            with open(path) as fh:
                records.append(RunRecord.model_validate(json.load(fh)))
        return records

    def completed_ids(self) -> set[str]:
        """Run ids that finished successfully. Failed runs are retried on resume."""
        if not self.runs_dir.exists():
            return set()
        done = set()
        for path in self.runs_dir.glob("*.json"):
            try:
                with open(path) as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") == "ok":
                done.add(payload["run_id"])
        return done

    def pending(self, specs: list[RunSpec]) -> list[RunSpec]:
        done = self.completed_ids()
        return [s for s in specs if s.run_id not in done]

    def rebuild_index(self) -> Path:
        """Regenerate index.csv from the run files. They are the source of truth."""
        self.root.mkdir(parents=True, exist_ok=True)
        rows = [r.flat_row() for r in self.all_records()]
        tmp = self.index_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, self.index_path)
        return self.index_path


def audit_completeness(record: RunRecord) -> list[str]:
    """Return the names of any axes a record failed to populate."""
    payload = record.model_dump(mode="json")
    return [axis for axis in REQUIRED_AXES if not payload.get(axis)]


def collect_environment(
    *,
    device: str,
    device_info: dict,
    clocks_locked_mhz: int | None,
    clock_lock_applied: bool,
) -> Environment:
    from . import __version__

    return Environment(
        device=device,
        device_info=device_info,
        clocks_locked_mhz=clocks_locked_mhz,
        clock_lock_applied=clock_lock_applied,
        torch_version=_version("torch"),
        transformers_version=_version("transformers"),
        kvpress_version=_version("kvpress"),
        platform=platform.platform(),
        git_sha=git_sha(),
        kvbench_version=__version__,
    )


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
