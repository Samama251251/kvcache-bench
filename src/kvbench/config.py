"""Experiment configuration and sweep expansion.

One YAML file describes a whole experiment. `ExperimentConfig.expand()` turns it
into the flat list of `RunSpec`s that make up the sweep -- one per experimental
cell. Each `RunSpec` hashes to a stable `run_id`, which is what makes `--resume`
work across sessions and machines.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MethodKind = Literal["full_cache", "eviction", "quantization"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    dtype: str = "bfloat16"
    attn_implementation: str | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cuda"
    hardware_backend: Literal["nvml", "fake"] = "nvml"
    device_index: int = 0
    power_sample_hz: float = 10.0
    warmup: bool = True
    # Runs are executed in a shuffled order so that thermal drift over a long
    # sweep does not correlate with method -- otherwise the last method to run
    # always looks like the hottest and slowest one.
    shuffle_seed: int = 1234
    # Optional `nvidia-smi -lgc` target. Recorded in every run record either way.
    lock_clocks_mhz: int | None = None


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = 128
    samples_per_task: int = 50
    # Greedy by default: sampling noise would show up as quality variance we
    # cannot attribute to the compression method.
    do_sample: bool = False


class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: MethodKind = "eviction"
    # Extra kwargs forwarded to the kvpress press constructor.
    params: dict = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: Literal["longbench", "ruler"]
    tasks: list[str]
    # RULER only: the synthetic context lengths to evaluate at.
    context_lengths: list[int] | None = None

    @model_validator(mode="after")
    def _check_context_lengths(self) -> BenchmarkConfig:
        if self.suite == "ruler" and not self.context_lengths:
            raise ValueError("ruler benchmarks must declare context_lengths")
        if self.suite != "ruler" and self.context_lengths:
            raise ValueError(f"context_lengths is only meaningful for ruler, not {self.suite}")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig
    methods: list[MethodConfig]
    benchmarks: list[BenchmarkConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    # Cache retention ratios for eviction methods. 1.0 belongs to the baseline
    # and is not swept over.
    budgets: list[float] = Field(default_factory=lambda: [0.5, 0.25, 0.1])
    # Bit widths for quantization methods.
    quant_bits: list[int] = Field(default_factory=list)
    repeats: int = 1

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if not any(m.kind == "full_cache" for m in self.methods):
            raise ValueError(
                "every sweep needs a full_cache baseline -- all reported numbers are relative to it"
            )
        names = [m.name for m in self.methods]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate method names: {names}")
        for b in self.budgets:
            if not 0.0 < b < 1.0:
                raise ValueError(f"budget {b} must be a retention ratio strictly between 0 and 1")
        if any(m.kind == "quantization" for m in self.methods) and not self.quant_bits:
            raise ValueError("a quantization method is configured but quant_bits is empty")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def expand(self) -> list[RunSpec]:
        """Flatten the config into every experimental cell, in declaration order."""
        specs: list[RunSpec] = []
        for method in self.methods:
            for budget, bits in _axis_values(method, self.budgets, self.quant_bits):
                for bench in self.benchmarks:
                    for task in bench.tasks:
                        for ctx in bench.context_lengths or [None]:
                            for repeat in range(self.repeats):
                                specs.append(
                                    RunSpec(
                                        experiment=self.name,
                                        model_id=self.model.id,
                                        dtype=self.model.dtype,
                                        method=method.name,
                                        method_kind=method.kind,
                                        method_params=method.params,
                                        retention=budget,
                                        quant_bits=bits,
                                        suite=bench.suite,
                                        task=task,
                                        context_length=ctx,
                                        repeat=repeat,
                                        max_new_tokens=self.generation.max_new_tokens,
                                        samples=self.generation.samples_per_task,
                                    )
                                )
        return specs


def _axis_values(
    method: MethodConfig, budgets: list[float], quant_bits: list[int]
) -> list[tuple[float | None, int | None]]:
    """The compression axis a method is swept along: retention ratio or bit width."""
    if method.kind == "full_cache":
        return [(1.0, None)]
    if method.kind == "quantization":
        return [(None, bits) for bits in quant_bits]
    return [(b, None) for b in budgets]


class RunSpec(BaseModel):
    """One cell of the sweep. Immutable, and hashes to a stable id."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    experiment: str
    model_id: str
    dtype: str
    method: str
    method_kind: MethodKind
    method_params: dict = Field(default_factory=dict)
    retention: float | None
    quant_bits: int | None
    suite: str
    task: str
    context_length: int | None
    repeat: int
    max_new_tokens: int
    samples: int

    @property
    def compression_ratio(self) -> float | None:
        """What kvpress presses actually take: the fraction of cache thrown away."""
        if self.retention is None:
            return None
        return round(1.0 - self.retention, 6)

    @property
    def run_id(self) -> str:
        """Readable slug plus a hash over the full spec, so ids stay unique."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
        return f"{self.slug}-{digest}"

    @property
    def slug(self) -> str:
        if self.quant_bits is not None:
            axis = f"{self.quant_bits}bit"
        else:
            axis = f"r{int(round((self.retention or 1.0) * 100)):03d}"
        parts = [self.method, axis, self.suite, self.task]
        if self.context_length is not None:
            parts.append(f"{self.context_length // 1024}k")
        parts.append(f"rep{self.repeat}")
        return _sanitize("-".join(parts))


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)
