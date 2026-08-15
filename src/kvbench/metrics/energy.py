"""Energy: the axis nobody else reports, so the one most worth getting right.

Joules come from trapezoidal integration of the sampled power curve over the run
window. Everything needed to audit that number after the fact -- sample count,
achieved rate, failed reads -- is carried alongside it, because a J/token figure
with no provenance is not evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .sampler import DeviceTrace


class EnergyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_joules: float
    joules_per_generated_token: float | None
    mean_watts: float
    peak_watts: float
    idle_watts: float | None
    # Joules above an idle baseline, when one was measured. The raw total is
    # still the headline; this is for asking how much of the draw was the work.
    net_joules_above_idle: float | None
    duration_s: float
    sample_count: int
    requested_hz: float
    achieved_hz: float
    read_errors: int

    @classmethod
    def from_trace(
        cls,
        trace: DeviceTrace,
        generated_tokens: int,
        idle_watts: float | None = None,
    ) -> EnergyMetrics:
        joules = integrate_joules(trace.seconds, trace.watts)
        duration = trace.duration_s
        net = None
        if idle_watts is not None and duration > 0:
            net = joules - idle_watts * duration
        return cls(
            total_joules=joules,
            joules_per_generated_token=(joules / generated_tokens if generated_tokens > 0 else None),
            mean_watts=(joules / duration if duration > 0 else 0.0),
            peak_watts=(max(trace.watts) if trace.watts else 0.0),
            idle_watts=idle_watts,
            net_joules_above_idle=net,
            duration_s=duration,
            sample_count=trace.sample_count,
            requested_hz=trace.requested_hz,
            achieved_hz=trace.achieved_hz,
            read_errors=trace.read_errors,
        )


def integrate_joules(seconds: list[float], watts: list[float]) -> float:
    """Trapezoidal integral of watts over seconds.

    Fewer than two samples means we never observed an interval, so the honest
    answer is zero energy rather than an extrapolation from a single reading.
    """
    if len(seconds) != len(watts):
        raise ValueError("timestamp and power series must be the same length")
    if len(seconds) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(seconds)):
        dt = seconds[i] - seconds[i - 1]
        if dt < 0:
            raise ValueError("timestamps must be non-decreasing")
        total += 0.5 * (watts[i] + watts[i - 1]) * dt
    return total


def measure_idle_watts(backend, seconds: float = 5.0, hz: float = 10.0) -> float:
    """Mean board power with nothing running, for the idle baseline."""
    from .sampler import DeviceSampler

    sampler = DeviceSampler(backend, hz=hz)
    sampler.start()
    sampler.wait(seconds)
    trace = sampler.stop()
    if not trace.watts:
        return 0.0
    return sum(trace.watts) / len(trace.watts)
