# Hardware and software behind every record in this directory

All 345 records were produced in one session on one card. Energy and latency
numbers are comparable within this directory and not with any other machine.

## GPU

| | |
|---|---|
| Card | NVIDIA RTX A6000 (Ampere GA102, compute capability 8.6) |
| VRAM | 48 GB GDDR6 (49140 MiB reported) |
| Power limit | 300 W, default and enforced, the card's maximum |
| Max clocks | 2100 MHz SM, 8001 MHz memory. Not locked; boost left on |
| Driver | 570.181 (CUDA 12.8 is the newest it supports) |
| VBIOS | 94.02.5C.00.02 |
| PCIe | Gen3 x16 slot, bus 03:00.0 |
| Idle draw | about 30 W at 300 MHz |

## Host

| | |
|---|---|
| Provider | Vast.ai, on-demand, host 64430, machine 10958, Romania |
| Instance | 49957031, unprivileged Docker container, not a VM |
| CPU | 2x Intel Xeon Gold 6248R @ 3.00 GHz. 48 threads on the host, 6 allocated to this container |
| RAM | 377 GB host, 48 GB allocated |
| Disk | 102 GB overlay on NVMe (QEMU virtual disk, ~3.7 GB/s) |
| Network | ~680 Mbps up, ~660 Mbps down |
| Price | $0.498/hr plus bandwidth |

## Software

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 5.15.0-139-generic |
| Image | Vast.ai PyTorch base image (github.com/vast-ai/base-image, derivatives/pytorch) |
| CUDA toolkit | 12.8 (nvcc 12.8.r12.8) |
| Python | 3.11.16 in the project venv |
| torch | 2.11.0+cu128, from download.pytorch.org/whl/cu128 |
| transformers | 5.2.0 |
| kvpress | 0.5.4 |
| optimum-quanto | 0.2.7 (backend for the quantized KV cache) |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| nvidia-ml-py | 13.610.43 (power sampling at 10 Hz) |
| Attention | sdpa for every method except H2O, which needs eager |
| kvcache-bench | 0.1.0 at git sha b78e9e30 for the sweep records |

## Model

`unsloth/Llama-3.1-8B-Instruct`, snapshot 4699cc75b550f9c6f3173fb80f4703b62d946aa5, bf16, 15 GB on disk. Same safetensors as `meta-llama/Llama-3.1-8B-Instruct`; the mirror was used because Meta's gate approval was pending.

## Datasets

`Xnhyacinth/LongBench` (tasks narrativeqa, qasper, hotpotqa, 2wikimqa, gov_report, triviaqa) and `simonjegou/ruler` at 4096, 8192 and 16384 tokens (niah_single_1, niah_multikey_1). Both are kvpress's preprocessed copies. RULER on the Hub has no 32k split.

## Measurement notes

- Clocks were not locked. The card ran at its 300 W limit with boost on, so power readings include boost behaviour. Repeat-to-repeat J/token variation across the sweep: median 1.2%, worst 5.2%.
- Peak memory at 16k, batch 1, full cache: 22.1 GB. H2O at 8k: 37 GB. H2O at 16k did not fit.
- The box's own idle draw is not subtracted from any number.
