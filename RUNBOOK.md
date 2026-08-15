# Runbook: one trip to the GPU

Written to be followed literally on the rented box. The aim is that the session
discovers facts about the GPU and nothing else.

## 0. Before you rent

```sh
uv run kvbench validate configs/full_sweep.yaml   # cell count + excluded cells
uv run kvbench plan configs/full_sweep.yaml --hourly-usd <price>
```

Pick the instance from that number. Prefer **several identical cards in one box**
over one card: the sweep shards across them at the same total GPU-hours and a
fraction of the wall-clock. Never two different card models — energy and latency
are not comparable across hardware, and the headline table must come from one
card type.

## 1. Set up the box

```sh
git clone <your remote> kvcache-bench && cd kvcache-bench
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
uv run kvbench doctor --config configs/full_sweep.yaml   # must print no DROP
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv
```

All cards must report the **same name**. If they do not, stop — that box cannot
produce one comparable table.

Pre-download the weights and data once, so 4 workers do not race the hub:

```sh
export HF_TOKEN=<token>          # Llama-3.1 is gated
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
m='meta-llama/Llama-3.1-8B-Instruct'
AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"
uv run python -c "
from datasets import load_dataset
for t in ['narrativeqa','qasper','hotpotqa','2wikimqa','gov_report','triviaqa']:
    load_dataset('Xnhyacinth/LongBench', data_dir=t, split='test')
for c in ['4096','16384','32768']:
    load_dataset('simonjegou/ruler', data_dir=c, split='test')"
```

## 2. Lock the clocks

Unlocked clocks let the card boost differently from run to run, which lands
directly in the energy numbers. Pick a value at or below the card's base clock
and put it in the config's `runtime.lock_clocks_mhz`:

```sh
sudo nvidia-smi -pm 1
nvidia-smi --query-gpu=clocks.max.sm --format=csv
```

If locking is refused (it needs privileges a rented box may not grant), the run
still proceeds and every record says `clock_lock_applied: false`. That is a
limitation to state in the write-up, not a reason to stop.

## 3. Set up off-box sync

**The single biggest risk in this whole plan is that the instance disappears
with the results on it.** On-box atomic writes do not help when the box is what
you lose. Point `--sync-cmd` at something that pushes each record off the
machine as it lands:

```sh
git config user.email you@example.com && git config user.name "you"
export SYNC='git add results && git commit -q -m "results: $(date -u +%FT%TZ)" || true; git push -q origin HEAD || true'
```

Anything that copies off-box works — `rclone`, `aws s3 sync`, `rsync` to your
laptop. A failing sync prints a warning and never kills the sweep.

## 4. Smoke session first

Do not start the full sweep. This is ~20 minutes and it is what stops you
finding out at hour 18 that something was wrong the whole time.

```sh
uv run kvbench run configs/smoke.yaml --sync-cmd "$SYNC"
```

Then read the records before going further:

```sh
uv run python -c "
import pandas as pd; d = pd.read_csv('results/index.csv')
print(d[['method','retention','primary_score','joules_per_generated_token','ttft_s','driver_peak_bytes','warnings']])
g = d[d.status=='ok'].groupby(['method','retention'])['joules_per_generated_token']
print((g.std()/g.mean()*100).rename('J/token spread %'))"
```

Four things to check, in order of how badly they matter:

1. **`hardware_backend` is `nvml`, not `fake`.** If it says fake, nothing you
   are about to run is a measurement.
2. **The J/token spread across the 3 repeats.** This is the noise floor. Under
   ~5% is good. If it is 15%+, differences smaller than that are not findings,
   and you should raise `repeats` or lock clocks before spending 20 hours.
3. **`warnings` is empty.** Thin power traces or degraded sample rates here mean
   the energy numbers will be soft everywhere.
4. **Peak memory at your longest context.** If 32k is near the card's limit,
   the full sweep will OOM on exactly the cells you care about most.

Then re-plan — the estimate switches from guessed to measured once records exist:

```sh
uv run kvbench plan configs/full_sweep.yaml --hourly-usd <price>
```

## 5. The full sweep

Under tmux, so an SSH drop does not end a 20-hour run. One worker per GPU:

```sh
tmux new -s sweep
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i uv run kvbench run configs/full_sweep.yaml \
    --shard $i/4 --device-index $i --sync-cmd "$SYNC" \
    > logs/worker$i.log 2>&1 &
done
wait
```

Workers shard a shuffled list, so each gets an interleaved mix of methods. That
matters: if one card in the box runs hotter than its neighbours, it shows up as
noise spread across all methods rather than as a bias attached to whichever
method happened to land on it. Every record stores its `device_index` so you can
test for a per-card effect afterwards instead of assuming there is none.

Detach with `Ctrl-B D`. Check in with:

```sh
tail -f logs/worker0.log
uv run kvbench plan configs/full_sweep.yaml   # cells done vs pending
```

## 6. If something goes wrong

Everything is resumable, and the mechanism is the same in every case:

```sh
uv run kvbench run configs/full_sweep.yaml --shard $i/4 --device-index $i --sync-cmd "$SYNC"
```

Completed cells are skipped, failed cells are retried. Specifically:

- **Instance preempted / SIGTERM.** The worker finishes the run in flight,
  writes it, syncs, and exits. Nothing in flight is lost.
- **A cell OOMs.** It is recorded with `status: failed` and the sweep continues.
  Retry it alone with `--limit`, or accept it and document it.
- **The box dies entirely.** Rent another of the **same card model**, clone,
  `uv sync --extra gpu`, and resume. The records you synced are already safe.
  Never finish a sweep on a different card model — start over or report two
  separate tables.

## 7. Before you release the box

```sh
uv run kvbench plan configs/full_sweep.yaml   # must read 0 pending
uv run python -c "
import pandas as pd; d = pd.read_csv('results/index.csv')
print(d.status.value_counts()); print('GPUs:', d.gpu.unique())
print('warned runs:', (d.warnings.notna() & (d.warnings != '')).sum())"
git add results && git commit -m "results: full sweep" && git push
```

`d.gpu.unique()` must be a single value. Then pull the results down and check
they are really on your machine before you destroy the instance.
