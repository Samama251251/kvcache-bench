"""Experiment configuration and sweep expansion.

One YAML file describes a whole experiment. `ExperimentConfig.expand()` turns it
into the flat list of `RunSpec`s that make up the sweep -- one per experimental
cell. Each `RunSpec` hashes to a stable `run_id`, which is what makes `--resume`
work across sessions and machines.

A sweep runs in *passes*, because quality and performance want opposite things
from the same generation:

  - quality only needs the text, so prompts can be batched and the run is 3-4x
    cheaper for identical scores;
  - energy and latency are only meaningful at batch size 1, but need far fewer
    samples, and are cleanest at a controlled context length.

Measuring both at once means paying batch-1 prices for every quality sample and
averaging energy over prompts of wildly varying length. Splitting them is both
faster and a better measurement.
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
PassMode = Literal["quality", "performance"]


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
    # Total prompt tokens allowed in one batch, across the whole batch. Caps the
    # batch size at long context so a 32k cell does not OOM a card that handles
    # the same batch size fine at 4k. Sized for a single >=40GB card.
    max_batch_tokens: int = 65536


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = 128
    # Default prompt budget; benchmarks may override it. Prompts over this are
    # truncated from the middle, keeping the head and the tail -- the same thing
    # LongBench's own harness does, and the tail is where the question lives.
    max_context_tokens: int | None = None
    # Greedy by default: sampling noise would show up as quality variance we
    # cannot attribute to the compression method.
    do_sample: bool = False


class EstimateConfig(BaseModel):
    """Rough throughput assumptions, used only until real records exist.

    These are declared guesses, not measurements -- `kvbench plan` says so in its
    output, and switches to measured per-run times as soon as the store has any.
    """

    model_config = ConfigDict(extra="forbid")

    prefill_tokens_per_s: float = 8000.0
    decode_tokens_per_s: float = 40.0
    # Used for suites whose prompt length is not known until the data is loaded.
    assumed_prompt_tokens: int = 8000
    # Fixed per-run cost: model already resident, but scoring and setup are not free.
    fixed_overhead_s: float = 20.0
    hourly_usd: float | None = None


class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: MethodKind = "eviction"
    # Extra kwargs forwarded to the kvpress press constructor.
    params: dict = Field(default_factory=dict)
    # Longest context this method can be run at. Cells above it are dropped from
    # the sweep rather than attempted and OOM'd. Observed-attention needs eager
    # attention, whose score matrix is quadratic in context: at 16k it asked a
    # 48GB card for 29GB more than it had.
    max_context_tokens: int | None = None


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: Literal["longbench", "ruler"]
    tasks: list[str]
    # RULER only: the synthetic context lengths to evaluate at.
    context_lengths: list[int] | None = None
    # Prompt budget for this suite, overriding generation.max_context_tokens.
    max_context_tokens: int | None = None

    @model_validator(mode="after")
    def _check_context_lengths(self) -> BenchmarkConfig:
        if self.suite == "ruler" and not self.context_lengths:
            raise ValueError("ruler benchmarks must declare context_lengths")
        if self.suite != "ruler" and self.context_lengths:
            raise ValueError(f"context_lengths is only meaningful for ruler, not {self.suite}")
        return self


class PassConfig(BaseModel):
    """One traversal of the method x budget x benchmark grid, for one purpose."""

    model_config = ConfigDict(extra="forbid")

    mode: PassMode
    samples_per_task: int
    # Quality passes batch; performance passes must not. Batching changes what
    # latency and energy mean, so batch_size > 1 is refused for performance.
    batch_size: int = 1
    repeats: int = 1
    # Restricts this pass to its own benchmark list. A performance pass wants
    # one controlled context length sweep, not the whole quality grid.
    benchmarks: list[BenchmarkConfig] | None = None

    @model_validator(mode="after")
    def _check(self) -> PassConfig:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.mode == "performance" and self.batch_size != 1:
            raise ValueError(
                "a performance pass must run at batch_size 1 -- batched latency and "
                "energy are not comparable to the batch-1 numbers papers report"
            )
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig
    methods: list[MethodConfig]
    benchmarks: list[BenchmarkConfig]
    passes: list[PassConfig] = Field(
        default_factory=lambda: [PassConfig(mode="quality", samples_per_task=50)]
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    estimate: EstimateConfig = Field(default_factory=EstimateConfig)
    # Cache retention ratios for eviction methods. 1.0 belongs to the baseline
    # and is not swept over.
    budgets: list[float] = Field(default_factory=lambda: [0.5, 0.25, 0.1])
    # Bit widths for quantization methods.
    quant_bits: list[int] = Field(default_factory=list)

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
        if not self.passes:
            raise ValueError("a sweep needs at least one pass")
        if any(m.kind == "eviction" for m in self.methods):
            batched = [p for p in self.passes if p.batch_size > 1]
            if batched:
                raise ValueError(
                    "eviction methods must run at batch_size 1: kvpress presses are not "
                    "padding-aware, so left-padded batches evict the wrong tokens "
                    "(StreamingLLM's sinks become pad tokens; SnapKV scores keys without "
                    "an attention mask). Batched at 4, SnapKV at 50% scored 15.7 on qasper "
                    "against a 49.9 baseline; at batch 1 it scored 48.3."
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def expand(self) -> list[RunSpec]:
        """Flatten the config into every experimental cell, in declaration order."""
        specs: list[RunSpec] = []
        for pass_config in self.passes:
            for method in self.methods:
                for budget, bits in _axis_values(method, self.budgets, self.quant_bits):
                    for bench in pass_config.benchmarks or self.benchmarks:
                        specs.extend(self._cells(pass_config, method, bench, budget, bits))
        return specs

    def _cells(
        self,
        pass_config: PassConfig,
        method: MethodConfig,
        bench: BenchmarkConfig,
        budget: float | None,
        bits: int | None,
    ) -> list[RunSpec]:
        budget_tokens = bench.max_context_tokens or self.generation.max_context_tokens
        cells = []
        for task in bench.tasks:
            for ctx in bench.context_lengths or [None]:
                context = ctx or budget_tokens
                if _exceeds_method_limit(method, context):
                    continue
                for repeat in range(pass_config.repeats):
                    cells.append(
                        RunSpec(
                            experiment=self.name,
                            model_id=self.model.id,
                            dtype=self.model.dtype,
                            mode=pass_config.mode,
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
                            max_context_tokens=budget_tokens,
                            samples=pass_config.samples_per_task,
                            batch_size=_batch_for(
                                pass_config.batch_size, context, self.runtime.max_batch_tokens
                            ),
                        )
                    )
        return cells

    def dropped_cells(self) -> list[tuple[str, int]]:
        """(method, context length) pairs excluded by a method's context limit.

        Surfaced by `kvbench validate` so a method missing from a plot is a
        documented decision rather than a mystery.
        """
        dropped = []
        for method in self.methods:
            if method.max_context_tokens is None:
                continue
            for bench in self.benchmarks:
                budget_tokens = bench.max_context_tokens or self.generation.max_context_tokens
                for ctx in bench.context_lengths or [budget_tokens]:
                    if ctx and _exceeds_method_limit(method, ctx):
                        dropped.append((method.name, ctx))
        return sorted(set(dropped))


def _exceeds_method_limit(method: MethodConfig, context: int | None) -> bool:
    if method.max_context_tokens is None or context is None:
        return False
    return context > method.max_context_tokens


def _batch_for(requested: int, context: int | None, max_batch_tokens: int) -> int:
    """Shrink the batch at long context so one budget works at every length.

    Batch 8 at 4k and batch 8 at 32k differ by 8x in cache footprint; without
    this, a config that is comfortable on the short cells OOMs on the long ones.
    """
    if requested <= 1 or not context:
        return max(1, requested)
    return max(1, min(requested, max_batch_tokens // context))


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
    mode: PassMode
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
    max_context_tokens: int | None
    samples: int
    batch_size: int = 1

    @property
    def compression_ratio(self) -> float | None:
        """What kvpress presses actually take: the fraction of cache thrown away."""
        if self.retention is None:
            return None
        return round(1.0 - self.retention, 6)

    @property
    def measures_performance(self) -> bool:
        """Whether this run's latency and energy belong in a performance plot."""
        return self.mode == "performance" and self.batch_size == 1

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
        parts = [self.mode[:4], self.method, axis, self.suite, self.task]
        if self.context_length is not None:
            parts.append(f"{self.context_length // 1024}k")
        parts.append(f"rep{self.repeat}")
        return _sanitize("-".join(parts))


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)
