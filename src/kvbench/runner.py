"""The runner: everything that happens around one measured generation.

The order of operations here is the experiment protocol, and it is identical for
every method:

    load model -> warm up -> start sampling -> generate all samples -> stop
    sampling -> score -> write record

Sweep-level discipline lives here too. Runs are executed in a shuffled order so
that thermal drift over a long session does not correlate with method; clocks
are locked when asked and the outcome recorded either way; and every run writes
its record immediately, so a crash costs one cell rather than the session.
"""

from __future__ import annotations

import random
import traceback
from dataclasses import dataclass

from .benchmarks import loader, scoring
from .config import ExperimentConfig, MethodConfig, RunSpec
from .hardware.base import HardwareBackend
from .metrics.energy import EnergyMetrics
from .metrics.latency import LatencyMetrics, LatencyRecorder
from .metrics.memory import (
    MemoryMetrics,
    driver_peak_bytes,
    geometry_from_hf_config,
    kv_cache_bytes,
    reset_torch_peak,
    torch_peak_bytes,
)
from .metrics.sampler import DeviceSampler, DeviceTrace
from .registry import build_method
from .results import (
    MethodInfo,
    QualityMetrics,
    ResultStore,
    RunRecord,
    collect_environment,
    utc_now,
)

# Presses that score on observed attention weights need eager attention -- the
# fused kernels never materialise the attention matrix they read.
EAGER_ONLY_PRESSES = {"ObservedAttentionPress"}

# A power trace this short cannot be integrated into a believable energy figure.
MIN_POWER_SAMPLES = 10
# How far below the requested sampling rate we tolerate before saying so.
MIN_RATE_FRACTION = 0.5


@dataclass
class GenerationOutcome:
    predictions: list[str]
    # Both describe the last sample, since a KV cache is per-sequence. Summing
    # prompt lengths across samples would make the cache look many times bigger
    # than any cache that ever existed.
    prompt_tokens: int
    retained_tokens: int | None


class ModelSession:
    """Holds a loaded model across runs so a sweep pays for loading once.

    Reloads only when the model, dtype or attention implementation actually
    changes -- the last of those matters because observed-attention needs eager
    attention and the other methods should not be forced to pay for it.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._key: tuple | None = None
        self.model = None
        self.tokenizer = None

    def get(self, spec: RunSpec, attn_implementation: str | None):
        key = (spec.model_id, spec.dtype, attn_implementation)
        if key != self._key:
            self.model, self.tokenizer = self._load(spec, attn_implementation)
            self._key = key
        return self.model, self.tokenizer

    def _load(self, spec: RunSpec, attn_implementation: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"dtype": getattr(torch, spec.dtype)}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        model.to(self.device)
        model.eval()
        return model, tokenizer


class _TokenTimer:
    """A logits processor used purely as a per-step clock.

    transformers calls it once per decoding step, so the first call marks time
    to first token and the rest give inter-token intervals. On CUDA it
    synchronises before stamping; that costs a little on every step, but it
    costs the same on every method, and un-synchronised stamps would time kernel
    launches rather than work.
    """

    def __init__(self, recorder: LatencyRecorder, device: str) -> None:
        self._recorder = recorder
        self._sync = str(device).startswith("cuda")

    def __call__(self, input_ids, scores):
        if self._sync:
            import torch

            torch.cuda.synchronize()
        self._recorder.mark_token()
        return scores


def build_inputs(tokenizer, sample: loader.Sample, device: str, max_context_tokens: int | None):
    """Tokenise one prompt, middle-truncating if it is over budget.

    Head and tail are kept and the middle dropped, which is what LongBench's own
    harness does -- the tail holds the question, and cutting it would change the
    task rather than shorten it.
    """
    import torch

    encoded = tokenizer(sample.prompt(), return_tensors="pt")
    ids = encoded["input_ids"][0]

    if max_context_tokens is not None and ids.shape[0] > max_context_tokens:
        head = max_context_tokens // 2
        tail = max_context_tokens - head
        ids = torch.cat([ids[:head], ids[-tail:]])

    ids = ids.unsqueeze(0).to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def generate_samples(
    *,
    model,
    tokenizer,
    method,
    samples: list[loader.Sample],
    spec: RunSpec,
    device: str,
    recorder: LatencyRecorder,
) -> GenerationOutcome:
    import torch

    predictions: list[str] = []
    prompt_tokens = 0
    retained_tokens: int | None = None
    timer = _TokenTimer(recorder, device)

    for sample in samples:
        inputs = build_inputs(tokenizer, sample, device, spec.max_context_tokens)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        recorder.start_sample()
        with torch.no_grad(), method.apply(model):
            output = model.generate(
                **inputs,
                max_new_tokens=spec.max_new_tokens,
                do_sample=False,
                logits_processor=[timer],
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                **method.generate_kwargs(),
            )
        recorder.end_sample()

        generated = output.sequences[0][inputs["input_ids"].shape[-1] :]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
        retained_tokens = _retained_tokens(output, len(generated))

    return GenerationOutcome(predictions, prompt_tokens, retained_tokens)


def _retained_tokens(output, generated: int) -> int | None:
    """How many positions the cache actually held, minus what we just generated.

    This is what makes the *effective* compression ratio measurable rather than
    assumed: presses do not always hit the requested ratio exactly, because of
    sink tokens, window minimums and per-layer budget rounding.
    """
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        return None
    try:
        length = int(cache.get_seq_length())
    except (AttributeError, TypeError):
        return None
    return max(0, length - generated)


def run_one(
    spec: RunSpec,
    method_config: MethodConfig,
    *,
    session: ModelSession,
    backend: HardwareBackend,
    config: ExperimentConfig,
    idle_watts: float | None = None,
    clock_lock_applied: bool = False,
) -> RunRecord:
    started = utc_now()
    method = build_method(method_config, spec)

    attn = config.model.attn_implementation
    if method.press_class in EAGER_ONLY_PRESSES:
        attn = "eager"

    environment = collect_environment(
        device=config.runtime.device,
        device_info=backend.device_info().as_dict(),
        clocks_locked_mhz=config.runtime.lock_clocks_mhz,
        clock_lock_applied=clock_lock_applied,
    )

    try:
        model, tokenizer = session.get(spec, attn)
        samples = loader.load_samples(
            spec.suite,
            spec.task,
            context_length=spec.context_length,
            limit=spec.samples,
        )
        if config.runtime.warmup and samples:
            _warmup(model, tokenizer, method, samples[0], spec, config.runtime.device)

        reset_torch_peak(config.runtime.device)
        recorder = LatencyRecorder()
        sampler = DeviceSampler(backend, hz=config.runtime.power_sample_hz)
        sampler.start()
        try:
            outcome = generate_samples(
                model=model,
                tokenizer=tokenizer,
                method=method,
                samples=samples,
                spec=spec,
                device=config.runtime.device,
                recorder=recorder,
            )
        finally:
            trace = sampler.stop()

        latency = recorder.metrics()
        energy = EnergyMetrics.from_trace(trace, latency.generated_tokens, idle_watts)
        memory = _memory_metrics(model, spec, outcome, trace, config.runtime.device)
        metric, primary, scores = scoring.score(spec.suite, spec.task, samples, outcome.predictions)
        quality = QualityMetrics(
            primary_metric=metric,
            primary_score=primary,
            scores=scores,
            samples_scored=len(samples),
        )
        status, error = "ok", None
        warnings = audit_measurement(trace, latency.generated_tokens)
    except Exception as exc:  # noqa: BLE001 - one bad cell must not end the sweep
        status, error = "failed", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        latency, energy, memory, quality = _empty_metrics(spec)
        outcome = GenerationOutcome([], 0, None)
        warnings = []

    return RunRecord(
        run_id=spec.run_id,
        slug=spec.slug,
        spec=spec.model_dump(mode="json"),
        method=MethodInfo(
            **method.describe(),
            effective_compression_ratio=_effective_ratio(memory),
        ),
        environment=environment,
        status=status,
        error=error,
        warnings=warnings,
        started_at=started,
        finished_at=utc_now(),
        quality=quality,
        memory=memory,
        latency=latency,
        energy=energy,
    )


def audit_measurement(trace: DeviceTrace, generated_tokens: int) -> list[str]:
    """Flag a run whose instrumentation underperformed, without failing it.

    A run that quietly produced a bad energy number is more dangerous than one
    that crashed: it looks like data.
    """
    warnings = []
    if trace.sample_count < MIN_POWER_SAMPLES:
        warnings.append(
            f"only {trace.sample_count} power samples over {trace.duration_s:.3f}s; "
            "energy is not measurable at this run length"
        )
    elif trace.achieved_hz < trace.requested_hz * MIN_RATE_FRACTION:
        warnings.append(
            f"power sampled at {trace.achieved_hz:.1f}Hz against a requested {trace.requested_hz:.1f}Hz"
        )
    if trace.read_errors:
        warnings.append(f"{trace.read_errors} failed device reads during the run")
    if generated_tokens == 0:
        warnings.append("no tokens were generated; per-token metrics are undefined")
    return warnings


def _warmup(model, tokenizer, method, sample, spec: RunSpec, device: str) -> None:
    """One unmeasured generation, so the first measured run is not the cold one."""
    import torch

    inputs = build_inputs(tokenizer, sample, device, spec.max_context_tokens)
    with torch.no_grad(), method.apply(model):
        model.generate(
            **inputs,
            max_new_tokens=min(8, spec.max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            **method.generate_kwargs(),
        )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _memory_metrics(
    model, spec: RunSpec, outcome: GenerationOutcome, trace: DeviceTrace, device: str
) -> MemoryMetrics:
    geometry = geometry_from_hf_config(model.config)
    total = outcome.prompt_tokens
    retained = outcome.retained_tokens
    if retained is None:
        # Fall back to the requested budget, which is what papers quote when
        # they do not measure the cache directly.
        retained = int(round(total * (spec.retention if spec.retention is not None else 1.0)))

    cache_dtype = f"int{spec.quant_bits}" if spec.quant_bits else spec.dtype
    return MemoryMetrics(
        torch_peak_bytes=torch_peak_bytes(device),
        driver_peak_bytes=driver_peak_bytes(trace),
        kv_cache_bytes=kv_cache_bytes(tokens=retained, dtype=cache_dtype, **geometry),
        kv_cache_bytes_uncompressed=kv_cache_bytes(tokens=total, dtype=spec.dtype, **geometry),
        retained_tokens=retained,
        prompt_tokens=total,
    )


def _effective_ratio(memory: MemoryMetrics) -> float | None:
    if memory.prompt_tokens == 0:
        return None
    return round(1.0 - memory.retained_tokens / memory.prompt_tokens, 6)


def _empty_metrics(spec: RunSpec):
    latency = LatencyMetrics(
        ttft_s=None,
        itl_mean_s=None,
        itl_p50_s=None,
        itl_p95_s=None,
        wall_s=0.0,
        generated_tokens=0,
        throughput_tokens_per_s=None,
    )
    energy = EnergyMetrics.from_trace(DeviceTrace(), 0)
    memory = MemoryMetrics(
        torch_peak_bytes=None,
        driver_peak_bytes=None,
        kv_cache_bytes=0,
        kv_cache_bytes_uncompressed=0,
        retained_tokens=0,
        prompt_tokens=0,
    )
    quality = QualityMetrics(
        primary_metric=scoring.metric_name(spec.suite, spec.task),
        primary_score=None,
        scores={},
        samples_scored=0,
    )
    return latency, energy, memory, quality


def run_sweep(
    config: ExperimentConfig,
    store: ResultStore,
    *,
    backend: HardwareBackend,
    resume: bool = True,
    limit: int | None = None,
    progress=None,
) -> list[RunRecord]:
    specs = config.expand()
    if resume:
        specs = store.pending(specs)

    # Shuffled, not grouped by method: a sweep that runs every SnapKV cell back
    # to back would hand SnapKV whatever thermal state the previous method left
    # behind, and that would look like a property of SnapKV.
    random.Random(config.runtime.shuffle_seed).shuffle(specs)
    if limit is not None:
        specs = specs[:limit]

    methods = {m.name: m for m in config.methods}
    session = ModelSession(config.runtime.device)

    clock_lock_applied = False
    if config.runtime.lock_clocks_mhz:
        clock_lock_applied = backend.lock_clocks(config.runtime.lock_clocks_mhz)

    records = []
    try:
        for index, spec in enumerate(specs, start=1):
            if progress:
                progress(index, len(specs), spec)
            record = run_one(
                spec,
                methods[spec.method],
                session=session,
                backend=backend,
                config=config,
                clock_lock_applied=clock_lock_applied,
            )
            store.save(record)
            records.append(record)
    finally:
        if clock_lock_applied:
            backend.reset_clocks()
    return records
