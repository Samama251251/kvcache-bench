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
from contextlib import contextmanager
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
from .registry import PRESS_CLASS_NAMES, build_method
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
    # Counted for every run, including batched ones where the latency recorder
    # is deliberately not fed. The count is a fact about the run; only the
    # per-token *timings* are meaningless under batching.
    generated_tokens: int = 0


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


def truncate_ids(ids, max_context_tokens: int | None):
    """Middle-truncate one token sequence if it is over budget.

    Head and tail are kept and the middle dropped, which is what LongBench's own
    harness does -- the tail holds the question, and cutting it would change the
    task rather than shorten it.
    """
    import torch

    if max_context_tokens is None or ids.shape[0] <= max_context_tokens:
        return ids
    head = max_context_tokens // 2
    tail = max_context_tokens - head
    return torch.cat([ids[:head], ids[-tail:]])


def build_batch(tokenizer, samples: list[loader.Sample], device: str, max_context_tokens: int | None):
    """Tokenise a batch, left-padded.

    Left padding, not right: a decoder-only model continues from the last
    position, so right padding would have it generating from pad tokens.
    """
    import torch

    # A chat template already carries the BOS token; adding another shifts every
    # position and the model notices.
    add_special = not loader.uses_chat_template(tokenizer)
    sequences = [
        truncate_ids(
            tokenizer(s.prompt(tokenizer), return_tensors="pt", add_special_tokens=add_special)["input_ids"][
                0
            ],
            max_context_tokens,
        )
        for s in samples
    ]
    width = max(int(s.shape[0]) for s in sequences)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        input_ids[row, width - seq.shape[0] :] = seq
        attention_mask[row, width - seq.shape[0] :] = 1

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, [int(s.shape[0]) for s in sequences]


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
    """Generate for every sample, in batches of `spec.batch_size`.

    The latency recorder is only fed on batch-1 runs. On a batched run a "token"
    is one step for the whole batch, so per-token timings would be a different
    quantity wearing the same name.
    """
    import torch

    predictions: list[str] = []
    prompt_tokens = 0
    generated_tokens = 0
    retained_tokens: int | None = None
    timing = spec.batch_size == 1
    processors = [_TokenTimer(recorder, device)] if timing else []

    for start in range(0, len(samples), spec.batch_size):
        chunk = samples[start : start + spec.batch_size]
        inputs, _ = build_batch(tokenizer, chunk, device, spec.max_context_tokens)
        # The padded width, not the unpadded length: the cache holds this many
        # positions, so it is what the retained count has to be compared against.
        width = int(inputs["input_ids"].shape[-1])
        prompt_tokens = width

        if timing:
            recorder.start_sample()
        with torch.no_grad(), method.apply(model):
            output = model.generate(
                **inputs,
                **generation_length(chunk, spec),
                do_sample=False,
                logits_processor=processors,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                **method.generate_kwargs(),
            )
        if timing:
            recorder.end_sample()

        generated = output.sequences[:, width:]
        generated_tokens += int(generated.shape[0] * generated.shape[-1])
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        retained_tokens = _retained_tokens(output, int(generated.shape[-1]))

    return GenerationOutcome(predictions, prompt_tokens, retained_tokens, generated_tokens)


def generation_length(chunk: list[loader.Sample], spec: RunSpec) -> dict:
    """How many tokens to generate, and whether that number is fixed.

    A quality run stops at end-of-turn like any deployment would. A performance
    run does not: it measures what a method costs per token, and an instruct
    model that answers a needle question in 7 tokens would have its prefill
    energy amortised over 7 tokens where a rambling method's is spread over 64.
    Pinning min to max makes every performance cell generate the same count, so
    J/token compares methods rather than answer lengths.
    """
    count = _new_tokens_for(chunk, spec)
    if spec.mode == "performance":
        return {"max_new_tokens": count, "min_new_tokens": count}
    return {"max_new_tokens": count}


def _new_tokens_for(chunk: list[loader.Sample], spec: RunSpec) -> int:
    """The benchmark's own generation budget, capped by the config.

    LongBench ships a per-task budget -- gov_report wants 512 tokens where
    hotpotqa wants 32 -- and using it rather than one flat number is what keeps
    our scores comparable to published ones. Samples in a chunk come from one
    task, so they agree; max() is just defensive.
    """
    declared = [s.max_new_tokens for s in chunk if s.max_new_tokens]
    if not declared:
        return spec.max_new_tokens
    return min(spec.max_new_tokens, max(declared))


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
    method = None

    environment = collect_environment(
        device=config.runtime.device,
        device_info=backend.device_info().as_dict(),
        clocks_locked_mhz=config.runtime.lock_clocks_mhz,
        clock_lock_applied=clock_lock_applied,
    )

    try:
        # Inside the try: a method that will not even construct should cost its
        # own cell, not the rest of the sweep.
        method = build_method(method_config, spec)

        attn = config.model.attn_implementation
        if method.press_class in EAGER_ONLY_PRESSES:
            attn = "eager"

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
        energy = EnergyMetrics.from_trace(trace, outcome.generated_tokens, idle_watts)
        memory = _memory_metrics(model, spec, outcome, trace, config.runtime.device)
        metric, primary, scores = scoring.score(spec.suite, spec.task, samples, outcome.predictions)
        quality = QualityMetrics(
            primary_metric=metric,
            primary_score=primary,
            scores=scores,
            samples_scored=len(samples),
        )
        status, error = "ok", None
        warnings = audit_measurement(trace, outcome.generated_tokens, spec)
    except Exception as exc:  # noqa: BLE001 - one bad cell must not end the sweep
        status, error = "failed", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        latency, energy, memory, quality = _empty_metrics(spec)
        outcome = GenerationOutcome([], 0, None, 0)
        warnings = []

    return RunRecord(
        run_id=spec.run_id,
        slug=spec.slug,
        spec=spec.model_dump(mode="json"),
        method=MethodInfo(
            **(method.describe() if method else _method_info_from_config(method_config, spec)),
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


def audit_measurement(trace: DeviceTrace, generated_tokens: int, spec: RunSpec) -> list[str]:
    """Flag a run whose instrumentation underperformed, without failing it.

    A run that quietly produced a bad energy number is more dangerous than one
    that crashed: it looks like data.
    """
    warnings = []
    if not spec.measures_performance:
        warnings.append(
            f"{spec.mode} pass at batch size {spec.batch_size}; latency and energy here are "
            "not comparable to batch-1 measurements and belong in no performance plot"
        )
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

    inputs, _ = build_batch(tokenizer, [sample], device, spec.max_context_tokens)
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


def _method_info_from_config(method_config: MethodConfig, spec: RunSpec) -> dict:
    """Describe a method that never got built, so the failure still has a name."""
    return {
        "name": method_config.name,
        "kind": method_config.kind,
        "press_class": PRESS_CLASS_NAMES.get(method_config.name),
        "requested_compression_ratio": spec.compression_ratio,
        "quant_bits": spec.quant_bits,
        "params": dict(method_config.params),
    }


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
    on_record=None,
) -> list[RunRecord]:
    methods = {m.name: m for m in config.methods}
    specs = config.expand()
    if resume:
        specs = store.pending(specs)

    specs = order_runs(specs, methods, config.runtime.shuffle_seed)
    if limit is not None:
        specs = specs[:limit]

    session = ModelSession(config.runtime.device)

    clock_lock_applied = False
    if config.runtime.lock_clocks_mhz:
        clock_lock_applied = backend.lock_clocks(config.runtime.lock_clocks_mhz)

    stop = _StopRequest()
    records = []
    try:
        with stop.installed():
            for index, spec in enumerate(specs, start=1):
                if stop.requested:
                    break
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
                if on_record:
                    on_record(record)
    finally:
        if clock_lock_applied:
            backend.reset_clocks()
    return records


def order_runs(specs: list[RunSpec], methods: dict, seed: int) -> list[RunSpec]:
    """Shuffle within attention-implementation groups, not across them.

    Two competing concerns. Shuffling matters: a sweep that runs every SnapKV
    cell back to back hands SnapKV whatever thermal state the previous method
    left behind, and that reads as a property of SnapKV. But observed-attention
    forces eager attention, and interleaving it with the rest would reload the
    model on almost every cell -- tens of minutes of a long sweep spent loading
    weights. Grouping by attention implementation and shuffling inside each
    group keeps the ordering protection where it can actually apply.
    """
    groups: dict[str | None, list[RunSpec]] = {}
    for spec in specs:
        method = methods.get(spec.method)
        eager = method is not None and PRESS_CLASS_NAMES.get(method.name) in EAGER_ONLY_PRESSES
        groups.setdefault("eager" if eager else None, []).append(spec)

    rng = random.Random(seed)
    ordered: list[RunSpec] = []
    for key in sorted(groups, key=lambda k: (k is not None, k or "")):
        group = groups[key]
        rng.shuffle(group)
        ordered.extend(group)
    return ordered


class _StopRequest:
    """Turns SIGTERM/SIGINT into 'stop after this run' instead of 'die now'.

    Preemption on a rented box arrives as SIGTERM. Without this it lands
    mid-generation and that cell's record is never written; with it the run in
    flight finishes, gets saved and synced, and the sweep exits clean for
    --resume to pick up.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous: dict = {}

    def _handle(self, signum, frame) -> None:
        if self.requested:
            # Asked twice: the operator means it.
            raise KeyboardInterrupt("second interrupt; stopping immediately")
        self.requested = True
        print(f"\nsignal {signum} received; finishing the current run, then stopping", flush=True)

    @contextmanager
    def installed(self):
        import signal

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except ValueError:
                # Not on the main thread; nothing to install.
                pass
        try:
            yield self
        finally:
            for sig, handler in self._previous.items():
                signal.signal(sig, handler)
