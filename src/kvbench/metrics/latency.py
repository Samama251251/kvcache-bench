"""Latency: where each family of methods pays for itself.

TTFT is the prefill number, so it is where prompt-phase compression (SnapKV,
PyramidKV) does its work. Inter-token latency is the decode number, so it is
where per-step eviction (TOVA, observed-attention) shows up. Reporting only
throughput would average the two together and hide exactly the difference we
are trying to measure.
"""

from __future__ import annotations

import time
from statistics import mean

from pydantic import BaseModel, ConfigDict


class LatencyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_s: float | None
    itl_mean_s: float | None
    itl_p50_s: float | None
    itl_p95_s: float | None
    wall_s: float
    generated_tokens: int
    throughput_tokens_per_s: float | None


class LatencyRecorder:
    """A stopwatch fed by generation callbacks.

    Accumulates across every sample in a run, so one record covers the whole
    task rather than a single prompt.
    """

    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self._ttfts: list[float] = []
        self._itls: list[float] = []
        self._wall = 0.0
        self._tokens = 0
        self._sample_start: float | None = None
        self._last_token_at: float | None = None

    def start_sample(self) -> None:
        self._sample_start = self._clock()
        self._last_token_at = None

    def mark_token(self) -> None:
        """Call once per generated token, in order."""
        now = self._clock()
        if self._sample_start is None:
            raise RuntimeError("mark_token() before start_sample()")
        if self._last_token_at is None:
            self._ttfts.append(now - self._sample_start)
        else:
            self._itls.append(now - self._last_token_at)
        self._last_token_at = now
        self._tokens += 1

    def end_sample(self) -> None:
        if self._sample_start is None:
            raise RuntimeError("end_sample() before start_sample()")
        self._wall += self._clock() - self._sample_start
        self._sample_start = None
        self._last_token_at = None

    def metrics(self) -> LatencyMetrics:
        return LatencyMetrics(
            ttft_s=(mean(self._ttfts) if self._ttfts else None),
            itl_mean_s=(mean(self._itls) if self._itls else None),
            itl_p50_s=_percentile(self._itls, 50),
            itl_p95_s=_percentile(self._itls, 95),
            wall_s=self._wall,
            generated_tokens=self._tokens,
            throughput_tokens_per_s=(self._tokens / self._wall if self._wall > 0 else None),
        )


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. No interpolation, so a value is always a real sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]
