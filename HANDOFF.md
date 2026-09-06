# Handoff: kvcache-bench after the first headline sweep

Written 2026-09-06 for whoever picks this up next, human or agent. Read this,
then `RUNBOOK.md`, then `results/HARDWARE.md` on the results branch. `idea.md`
is the original spec and is gitignored, so it only exists on the owner's machine.

## One paragraph

kvcache-bench measures KV cache compression methods on one model, one card, one
harness, and reports GPU energy per generated token next to quality, memory and
latency. On 2026-09-05 the full sweep ran on a rented RTX A6000: 345 cells,
zero failures, about 8.5 GPU-hours, about $4.50. Every record is committed on
the branch `results/a6000-2026-09-05`, which has not been merged into `main`
yet. The instrument works. The result is that eviction methods save memory and
almost no energy at 4k to 16k on an 8B model at batch 1, quantized cache costs
27 to 35% more energy per token, and H2O as kvpress implements it costs roughly
double. The report has not been written.

## Where everything is

| What | Where |
|---|---|
| Code, configs, tests, runbook | `main` at `1bf29f8` or later |
| The 345 sweep records | branch `results/a6000-2026-09-05`, directory `results/runs/*.json`, one file per cell |
| Flat index of all records | same branch, `results/index.csv`, one row per cell, 32 columns |
| Hardware and software spec of the run | same branch, `results/HARDWARE.md` |
| The spec | `idea.md`, local only, gitignored |
| Agent guidance and writing rules | `CLAUDE.md` (also gitignored, local only) and `AGENTS.md` |
| How a GPU trip goes | `RUNBOOK.md` |

The results branch is `main` plus 444 commits: one per record, a few merges of
`main`, and the HARDWARE.md commit. Merge it into `main` with a merge commit,
not a squash, or the per-record history that proves nothing was edited by hand
is lost. There are no conflicts; it already contains `main`.

The rented instance has been destroyed. Nothing that mattered lived only there.
A deploy key named `vast-a6000-2026-09-05` may still exist in the GitHub repo
settings and should be deleted.

## What the harness does

`kvbench` is one CLI, four commands:

```sh
uv run kvbench doctor --config configs/full_sweep.yaml   # which press classes resolve
uv run kvbench validate configs/full_sweep.yaml          # expand the grid, list dropped cells
uv run kvbench plan configs/full_sweep.yaml --hourly-usd 0.50   # GPU-hours and dollars
uv run kvbench run configs/full_sweep.yaml --sync        # run it, pushing each record to git
```

A config names a model, a list of methods, budgets, benchmarks and one or more
passes. `config.expand()` turns that into `RunSpec` cells. `runner.run_sweep`
runs them in a seeded shuffled order (eager-attention methods grouped so the
model reloads once), writes one JSON record per cell atomically, appends to
`index.csv`, and with `--sync` commits and pushes after every record. Rerunning
the same command skips completed cells and retries failed ones. SIGTERM finishes
the cell in flight, syncs, and exits.

Two passes over the same grid, because quality and performance want opposite
things from generation:

- **quality**: many samples, stops at end-of-turn like a deployment would,
  scored with the benchmark's own metric (LongBench F1/ROUGE-L via the vendored
  LongBench scorer, RULER substring match). Predictions and references are kept
  on the record.
- **performance**: few samples, three repeats, batch 1, and `min_new_tokens`
  pinned to `max_new_tokens` so every method generates the same count. This is
  where J/token, TTFT, inter-token latency, throughput and peak memory come
  from. Energy is an NVML power thread at 10 Hz integrated over the generation
  window.

Layout, so you know where to look:

```
src/kvbench/
  cli.py          the four commands
  config.py       pydantic config, grid expansion, the validators that refuse bad setups
  runner.py       model session, prompt building, generate loop, one cell end to end
  results.py      RunRecord schema, ResultStore (atomic writes, index, resume)
  sync.py         git add/commit/push per record
  estimate.py     GPU-hour estimate, declared or measured
  registry.py     method name -> adapter class
  benchmarks/     loader (HF snapshots, chat-template prompts), scoring, vendored kvpress scorers
  methods/        full_cache, kvpress_press (all five presses), quantized (HF quantized cache)
  metrics/        energy integration, latency recorder, memory, NVML sampler thread
  hardware/       nvml backend and a fake backend for the macOS dev path
tests/            138 tests, all pass on macOS with no GPU (uses a tiny random Llama)
configs/
  full_sweep.yaml the headline grid, exactly as run
  smoke.yaml      the one-hour gate you run first on any new card
  local_tiny.yaml laptop end-to-end check, fake hardware backend
```

Dev on macOS: `uv sync && uv run pytest -q`. Nothing CUDA runs locally; the fake
backend fabricates power so the pipeline is exercised end to end.

## What was run

Model `unsloth/Llama-3.1-8B-Instruct` (byte-identical mirror of Meta's repo),
bf16, sdpa attention. Card: RTX A6000 48GB, 300 W limit, clocks unlocked,
driver 570.181. Full details in `results/HARDWARE.md`.

Methods and budgets:

| Method | Adapter | Budgets |
|---|---|---|
| full_cache | no compression, the baseline | 100% |
| streaming_llm | kvpress `StreamingLLMPress` | 50 / 25 / 10% retention |
| h2o | kvpress `ObservedAttentionPress`, needs eager attention | 50 / 25 / 10%, 8k context cap |
| snapkv | kvpress `SnapKVPress` | 50 / 25 / 10% |
| pyramidkv | kvpress `PyramidKVPress` | 50 / 25 / 10% |
| tova | kvpress `TOVAPress` | 50 / 25 / 10% |
| kv_quant | transformers `cache_implementation="quantized"`, quanto backend | 4-bit, 2-bit |

Benchmarks: LongBench narrativeqa, qasper, hotpotqa, 2wikimqa, gov_report,
triviaqa (kvpress's `Xnhyacinth/LongBench` copy, 16k prompt budget,
middle-truncated); RULER niah_single_1 and niah_multikey_1 at 4k, 8k, 16k
(kvpress's `simonjegou/ruler` copy).

Passes: quality at 50 samples per task, batch 1; performance on niah_single_1
only at the three context lengths, 5 samples, 3 repeats, batch 1.

Grid: 345 cells. 192 quality, 153 performance. H2O has 8k and 4k cells only.

## Headline numbers

LB6 is the mean over the six LongBench tasks. RULER is mean needle accuracy over
4k/8k/16k. Energy, latency and memory are batch 1 at 16k, relative to baseline.

| Method | LB6 | RULER | J/token vs baseline | Inter-token ms | Peak GB |
|---|---|---|---|---|---|
| full_cache | 48.4 | 100 | 16.7 J | 40 | 22.1 |
| snapkv 50 / 25 / 10% | 47.8 / 47.4 / 45.7 | 100 / 100 / 100 | -2% / -6% / -8% | 40 | 20.0 / 19.0 / 18.4 |
| pyramidkv 50 / 25 / 10% | 48.0 / 47.2 / 45.7 | 100 / 100 / 100 | -3% / -5% / -8% | 40 | 20.0 / 19.0 / 18.4 |
| tova 50 / 25 / 10% | 48.0 / 46.6 / 45.2 | 100 / 100 / 99.7 | -5% / -6% / -9% | 40 | 20.0 / 19.0 / 18.4 |
| streaming_llm 50 / 25 / 10% | 45.0 / 42.6 / 40.7 | 53 / 30 / 11 | -4% / -5% / -8% | 41 | 20.0 / 19.0 / 18.4 |
| kv_quant 4-bit / 2-bit | 48.2 / 40.5 | 100 / 91 | +35% / +27% | 56 / 54 | 19.1 / 18.6 |
| h2o 50 / 25 / 10% (at 8k) | none | 74 / 31 / 12 | +97% / +84% / +76% | 47 to 55 | 37 |

Noise floor: J/token across the three repeats of one cell varies 1.2% at the
median and 5.2% at worst. Eviction differences at 50% are inside that. The 10%
savings, the quantization cost and the H2O cost are well outside it.

Findings, stated the way the report should state them:

1. At batch 1 on an 8B model, decode reads 16 GB of weights per token. A 16k
   cache is 2 GB. Evicting 90% of it saves 8 to 9% energy and no latency. The
   regime where compression papers evaluate is not the regime where compression
   pays for itself in energy.
2. The quanto quantized cache costs 27 to 35% more energy and 13 to 16 ms more
   per token for a 3 GB saving at 16k. 2-bit also loses 8 LongBench points.
3. kvpress's H2O needs eager attention. That doubles energy per token,
   quadruples time to first token at 8k, peaks at 37 GB, and cannot run at 16k
   on 48 GB. Its needle accuracy was also the worst of the attention-scored
   methods.
4. SnapKV, PyramidKV and TOVA are indistinguishable within noise on every axis.
   All hold needle retrieval at 100% down to 10% and lose 2 to 3 LongBench
   points at 10%, mostly on qasper.
5. StreamingLLM keeps LongBench within 3 to 8 points and collapses on retrieval.
6. kvpress presses are not padding-aware. Batched at 4 with left padding,
   SnapKV at 50% scored 15.7 on qasper against a 49.9 baseline and StreamingLLM
   5.6; at batch 1 they score 48.3 and 27.2. StreamingLLM's sinks are the first
   four positions, which are pad tokens in a padded row, and SnapKV scores keys
   with no attention mask. Anyone batching kvpress evaluations gets wrong
   numbers. This is worth a paragraph of its own.

## Deviations from the spec, and why

Each of these is a fact about the run that the report must state.

| Spec said | What ran | Why |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `unsloth/Llama-3.1-8B-Instruct` | Meta's gate approval was pending when the card was already billing. Same safetensors. |
| flash_attention_2 | sdpa | No flash-attn wheel for torch 2.11 cu128, the newest torch a 570 driver runs. A source build on 6 vCPUs is an hour. |
| RULER at 4k/16k/32k | 4k/8k/16k | kvpress's RULER copy on the Hub has no 32k split. |
| H2O to 16k | H2O capped at 8k | Eager attention asked for 29 GB more than the card had at 16k. Consequence: H2O has no LongBench cells, because that suite's budget is 16k. |
| Quality passes batched | batch 1 | Presses evict the wrong tokens under padding (finding 6). The config loader now refuses batched passes when an eviction method is present. |
| Not specified | Performance cells pin generation length | Otherwise J/token amortises prefill over however many tokens each method happened to emit before end-of-turn. |
| 4090 or A100 | A6000 48GB | Cheapest card that clears 40 GB. Only one card type appears in the records. |

## What broke on the way, so you do not repeat it

- **PyPI torch is built for CUDA 13.** Rented cards mostly run 570-series
  drivers that stop at 12.8. `pyproject.toml` now pins Linux to the cu128
  index. If `torch.cuda.is_available()` is False on a fresh box, this is why.
- **`uv sync` on a venv that had both cu13 and cu12 nvidia packages deleted
  shared files.** Symptom: `libcusparseLt.so.0: cannot open shared object`.
  Fix: `uv pip install --reinstall-package` for nvidia-cusparselt-cu12,
  nvidia-nvshmem-cu12, nvidia-cudnn-cu12, nvidia-nccl-cu12. A fresh venv from
  the current lock does not have this problem.
- **Unauthenticated Hub requests rate-limit after a few dozen**, and the
  datasets library then falls back to its cache under a mismatched config key.
  The loader now snapshots each task directory once and reads parquet from
  disk. Setting `HF_TOKEN` also helps.
- **transformers caches its optional-dependency checks at import.** Installing
  optimum-quanto into a running sweep does nothing until the process restarts.
  SIGTERM the process and rerun the same command.
- **`pkill -f` with a pattern that appears in your own ssh command kills your
  own shell**, and the tmux server if the pattern is in its command line. Use
  `pkill -f "kvbench run configs/full_swee[p]"` or target the PID.
- **Sync passed a relative results path to git while running from inside that
  directory.** Fixed; there is a regression test.
- Every smoke record from before these fixes was deleted from the results
  branch. Nothing in `results/` predates the fixed code.

## Records: what a row means

`results/index.csv` columns worth knowing: `run_id`, `mode` (quality or
performance), `method`, `method_kind`, `retention` (1.0 for baseline, empty for
quantization), `quant_bits`, `suite`, `task`, `context_length` (RULER only),
`repeat`, `status`, `warnings`, `primary_metric`, `primary_score`,
`torch_peak_bytes`, `kv_cache_bytes`, `ttft_s`, `itl_mean_s`,
`throughput_tokens_per_s`, `generated_tokens`, `total_joules`,
`joules_per_generated_token`, `mean_watts`, `peak_watts`, `gpu`.

Use `mode == "performance"` rows for anything about energy, latency or memory.
Quality rows carry energy too but generation length varies per sample, and
every quality row has a warning saying its latency and energy are not the
measurement. The full JSON record adds the environment (driver, torch,
transformers, kvpress versions, git sha), the power trace summary, and for
quality rows the predictions and references.

Aggregate over repeats with the mean; report the spread. Mean over LongBench
tasks is a fair summary because the tasks were chosen to span single-doc QA,
multi-doc QA, summarisation and few-shot QA. Do not average LongBench and RULER
together; they are different scales measuring different things.

## What is not done

1. **The report.** `idea.md` wants a long-form write-up: problem, methods,
   setup, trade-off curves, findings, limitations. Nothing exists yet. The
   numbers above and `results/HARDWARE.md` are the inputs. Tone rules are in
   `CLAUDE.md`: empirical, name where compression hurts, no puffery.
2. **Plots.** Nothing generates figures from `index.csv` yet. The obvious set:
   quality vs retention per method with the baseline as a line; J/token vs
   retention at each context length; peak memory vs retention; needle accuracy
   vs context length per method. Read the `dataviz` guidance before writing
   them if you are an agent with that skill.
3. **README.** Still a stub. Needs setup, methodology, the headline table, and
   the deviations table.
4. **Merge the results branch.** Merge commit, not squash.
5. **Delete the deploy key** from the GitHub repo settings.

## What would be worth a second GPU session

Only if the report needs it. Each is a config change, not code.

- **H2O on LongBench**: run all methods with LongBench truncated to 8k so H2O
  has comparable cells. About 2 hours on a 48 GB card. Without it the report
  can only compare H2O on retrieval.
- **A regime where the cache dominates**: 32k or 64k context, or batch 8 for
  the baseline and kv_quant only (eviction presses cannot batch). This is where
  eviction would start to save real energy. RULER has no 32k data on the Hub,
  so it means generating needle prompts locally or using a different long
  benchmark. This is a new experiment, not a fix to this one.
- **Locked clocks**: `runtime.lock_clocks_mhz` is plumbed but was left null.
  Locking would tighten the noise floor below 1.2%. Not needed for the current
  claims.
- **Qwen2.5-7B-Instruct**: the spec's stretch goal. Change `model.id`, rerun
  the smoke config, then the sweep. Zero code. kvpress supports it. Do not mix
  its numbers into the Llama tables.

## Things an agent should not do

- Do not batch quality passes with eviction methods. The loader refuses it;
  do not remove the check.
- Do not mix cards in one table. `results/HARDWARE.md` names the one card
  every record came from.
- Do not edit records by hand. Rerun the cell.
- Do not propose new compression methods, vLLM backends, multi-GPU, training
  or LLM-as-a-judge. All out of scope by the spec.
- Do not run `uv sync` on a box without checking `torch.cuda.is_available()`
  afterwards.
