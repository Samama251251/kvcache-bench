# Runbook: one trip to the GPU

For a single rented card with **≥40GB VRAM**. Take the code there once, run
everything, bring the results back.

The order matters. The smoke session exists so that the long sweep discovers
facts about the GPU rather than bugs, and so that you find out about a bad
measurement in an hour instead of after twenty.

---

## Before you rent

On your own machine, confirm the sweep is the size you think it is:

```sh
uv run kvbench validate configs/full_sweep.yaml
uv run kvbench plan configs/full_sweep.yaml --hourly-usd <price>
```

`validate` prints the per-pass breakdown and, at the bottom, any cells dropped
by a method's context limit. Those are documented decisions — if something you
expected to see is missing, deal with it now.

Push everything to a remote you can pull from on the box, and make sure you can
push *back* from it. That is what protects the results.

---

## 1. Set up the box (~1 hour)

```sh
git clone <your-remote> kvcache-bench && cd kvcache-bench
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
```

Flash attention is not in the lockfile because it builds from source and takes
20–40 minutes. Install it separately, and if it fails, set
`model.attn_implementation: sdpa` in the configs rather than losing the session
to a compiler error:

```sh
uv pip install flash-attn --no-build-isolation
```

Authenticate for the model download (Llama-3.1 is gated) and pre-fetch, so a
download failure happens now rather than at run 40:

```sh
export HF_TOKEN=<your token>
uv run huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

Confirm the harness agrees with the machine:

```sh
uv run kvbench doctor --config configs/full_sweep.yaml
nvidia-smi
```

Set up git so the box can push results back:

```sh
git config user.name "<you>" && git config user.email "<you>"
git checkout -b results/<card>-<date>
```

---

## 2. Smoke session (~1 hour)

```sh
tmux new -s smoke
uv run kvbench run configs/smoke.yaml --sync
```

Then **read the results before going further**. Four questions, in order of how
badly a wrong answer would hurt:

**Is energy measurable at all?** Every record should have a `sample_count` in
the hundreds and no `power samples` warning. If NVML is not reporting, the
project's headline metric does not exist on this box and nothing else matters.

```sh
uv run python -c "
import csv; rows=list(csv.DictReader(open('results/index.csv')))
for r in rows:
    if r['mode']=='performance':
        print(r['method'], r['context_length'], r['total_joules'], r['warnings'][:60])
"
```

**What is the energy noise floor?** The performance pass runs `repeats: 3`.
Compare J/token across the three identical runs of the same cell. If they vary
by more than a few percent, the differences this project wants to report between
methods are not falsifiable — say so in the write-up, and consider raising
repeats before the full sweep rather than after.

**Does 32k fit?** The smoke config runs RULER at 32768 deliberately. If it OOMs,
drop 32k from `full_sweep.yaml` now and run at 4k/8k/16k. Better a narrower
claim than a hole in the table.

**Can the model still do the task uncompressed at 16k?** Check the `full_cache`
RULER score. If the *uncompressed* baseline is already poor, compression-induced
degradation is floor-limited and unmeasurable — the headline result would be
meaningless. This is the check most worth not skipping.

Re-plan with real numbers before committing to the long run:

```sh
uv run kvbench plan configs/full_sweep.yaml --hourly-usd <price>
```

It switches from `declared` to `measured` once records exist. Trust that number,
not the pre-trip estimate.

---

## 3. Full sweep (~10–16 hours)

Detached, so an SSH drop does not end it:

```sh
tmux new -s sweep
uv run kvbench run configs/full_sweep.yaml --sync 2>&1 | tee sweep.log
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t sweep`.

`--sync` commits and pushes each record as it lands. That is the real protection:
atomic per-run writes survive the *process* dying, but only pushing survives the
*machine* dying, which on a preemptible box is the likelier of the two.

**If it stops for any reason**, the same command resumes it. Completed cells are
skipped, failed ones are retried:

```sh
uv run kvbench run configs/full_sweep.yaml --sync
```

SIGTERM (preemption, `docker stop`, `kill`) finishes the run in flight, saves it,
syncs, and exits cleanly. Pressing Ctrl-C twice stops immediately.

---

## 4. Before you release the box

```sh
uv run kvbench plan configs/full_sweep.yaml     # expect 0 pending
git add results && git commit -m "Add full sweep results" && git push
```

Then verify from **your own machine**, not the rented one, that the results
actually arrived:

```sh
git pull && ls results/runs | wc -l
uv run python -c "
import csv; rows=list(csv.DictReader(open('results/index.csv')))
print(len(rows), 'runs')
print('failed:', sum(1 for r in rows if r['status']!='ok'))
print('with warnings:', sum(1 for r in rows if r['warnings']))
print('cards:', {r['gpu'] for r in rows})
"
```

`cards` must be a single entry. Energy and latency are not comparable across
hardware, so two entries means two tables in the report, never one merged plot.

Only after that has succeeded, destroy the instance.

---

## What to do when things break

**CUDA OOM on one cell.** It fails that cell and the sweep continues. Check
which cells failed at the end; if it is the 32k ones, lower `max_batch_tokens`
or drop 32k and re-run — resume will only redo the failures.

**Flash attention will not install.** Set `attn_implementation: sdpa`. Slower,
but correct, and it applies equally to every method so the comparison holds.

**A press errors on every cell.** Stop, run `kvbench doctor`, and drop the
method from the config. A dropped method documented in the report beats a
half-populated column.

**Sync failures piling up.** The summary line at the end reports them. Records
are still on disk — commit and push manually before releasing the box.

**The run looks stuck.** 32k prefill on a batch is genuinely slow. Check
`nvidia-smi` for utilisation before assuming a hang.
