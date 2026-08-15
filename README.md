# kvcache-bench

An energy-aware, apples-to-apples benchmark of KV cache compression methods for
LLM inference.

Papers on KV cache compression each evaluate on different models, benchmarks,
budgets and hardware — and almost none of them report energy. This runs several
methods through **one harness** on **identical model, hardware, benchmarks and
budgets**, and reports **GPU energy (joules per generated token)** alongside
quality, memory and latency.

This is not a new compression method. It is the measurement instrument and the
comparison.

## What gets measured

Every run produces one record carrying all four axes. None of them is optional:
a half-populated record would silently drop rows out of the cross-cutting
comparisons that are the point of the project.

| Axis | Metrics |
| --- | --- |
| Quality | LongBench task-native scores (F1, ROUGE-L, classification, retrieval); RULER string match |
| Memory | torch allocator peak, driver peak, computed KV cache bytes |
| Speed | TTFT, inter-token latency (mean/p50/p95), throughput |
| Energy | total joules, **J per generated token**, mean and peak watts |

TTFT and ITL stay separate on purpose: prefill-phase methods (SnapKV, PyramidKV)
and per-step methods (TOVA, observed attention) pay their overhead in different
places, and throughput alone averages that away.

## Install

Everything runs through [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # dev machine; uses the fake hardware backend
uv sync --extra gpu      # on the GPU box; adds NVML
```

## Use

```sh
uv run kvbench doctor                      # which methods the installed kvpress can run
uv run kvbench validate configs/smoke.yaml # config parses and expands
uv run kvbench plan configs/full_sweep.yaml --hourly-usd 0.80
uv run kvbench run configs/smoke.yaml      # writes one record per run as it goes
```

One YAML plus one command reproduces any experiment. `run` resumes by default:
cells that already completed successfully are skipped, failed ones are retried.

For an actual GPU session, see **[RUNBOOK.md](RUNBOOK.md)** — the order of
operations matters, and the smoke session exists to catch things that would
otherwise waste the long run.

## Passes

A sweep runs the same grid twice, because quality and performance want opposite
things from the same generation:

- a **quality** pass batches prompts (8 at a time, fewer at long context) and
  measures scores. Batching is 3-4x cheaper for identical scores.
- a **performance** pass runs at batch size 1, few samples, repeated, at
  controlled context lengths. That is the only way latency and energy mean
  anything.

Measuring both at once would pay batch-1 prices for every quality sample *and*
average energy over prompts of wildly varying length. Records from a batched
pass carry an explicit warning that their latency and energy belong in no
performance plot, and a `performance` pass with `batch_size > 1` is rejected by
config validation.

## Methods

Method names map to NVIDIA [kvpress](https://github.com/NVIDIA/kvpress) presses.
Against kvpress 0.5.4 all of these resolve — `kvbench doctor` re-checks against
whatever is actually installed rather than trusting this table.

| Config name | Backing class |
| --- | --- |
| `full_cache` | none — the uncompressed baseline, a first-class method |
| `streaming_llm` | `StreamingLLMPress` |
| `h2o` | `ObservedAttentionPress` (forces eager attention) |
| `snapkv` | `SnapKVPress` |
| `pyramidkv` | `PyramidKVPress` |
| `tova` | `TOVAPress` |
| `knorm`, `expected_attention`, `random` | cheap reference points |
| `kv_quant` | transformers quantized cache — kvpress ships no KIVI-style press |

Budgets are **retention** ratios in configs (what you keep); presses take the
discarded fraction, and the conversion happens in one place. Quantization sweeps
bit widths instead.

## Measurement discipline

- **Never mix GPU types in one plot or table.** Energy and latency are not
  comparable across cards. Every record names the GPU and hardware backend it
  came from.
- **Records marked `hardware_backend: fake` are not measurements.** The fake
  backend exists so the harness can be developed and tested without a GPU.
- Runs execute in a **shuffled order**, not grouped by method, so thermal drift
  over a long session does not line up with method order.
- Each run does an **unmeasured warmup pass** first, so the first measured run
  is not the cold one.
- Clock locking is attempted when configured and the **outcome is recorded**,
  since it usually needs privileges a rented box may not grant.
- Runs carry `warnings` for thin power traces, degraded sampling rates, failed
  device reads and batched passes. A quietly bad number is worse than a crash.
- Methods can declare a `max_context_tokens`; cells above it are **dropped and
  reported** by `kvbench validate` rather than attempted. H2O
  (`ObservedAttentionPress`) needs eager attention, whose score matrix is
  quadratic in context -- roughly 17GB for one layer at 16k and 69GB at 32k, so
  its 32k cells cannot run on any single card.

## Results

`results/runs/<run_id>.json` is one self-describing record per run, written
atomically the moment that run finishes. `results/index.csv` is regenerated from
those files — the run files are the source of truth. Plots come from this data,
never from re-running GPUs.

## Layout

```
src/kvbench/
  config.py      YAML -> RunSpecs; passes, budgets, batch sizing, run ids
  registry.py    method name -> kvpress class; the Phase 0 gate as code
  runner.py      the experiment protocol, identical for every method
  results.py     record schema, atomic writes, resume
  estimate.py    GPU-hours and cost before renting anything
  sync.py        pushes each record off the box as it lands
  methods/       one adapter per family; no per-method scripts
  hardware/      NVML and fake backends behind one interface
  metrics/       energy, memory, latency, background sampler
  benchmarks/    LongBench and RULER loading and scoring
```

## Development

```sh
uv run pytest
uv run ruff check src/ tests/
```

The test suite runs a real sweep end to end on this machine: a tiny Llama on
CPU, with the fake power backend and stand-in benchmark data. It needs the
HuggingFace hub once to fetch the model, and skips itself if the hub is
unreachable.

The scorers in `src/kvbench/benchmarks/vendored/` are copied verbatim from
kvpress and must stay byte-identical — see the README there.
