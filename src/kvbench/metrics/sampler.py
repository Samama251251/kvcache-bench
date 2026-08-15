"""Background device sampling.

One thread samples both board power and device memory on a fixed cadence for the
duration of a run. Power is the signal we integrate into joules; memory gives us
a driver-level high-water mark to cross-check against torch's allocator view.

Sampling in a thread rather than around each generate() call is what makes the
energy number method-agnostic: nothing in a compression adapter can influence
what gets recorded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..hardware.base import HardwareBackend


@dataclass
class DeviceTrace:
    """Raw samples for one run window. Seconds are relative to sampler start."""

    seconds: list[float] = field(default_factory=list)
    watts: list[float] = field(default_factory=list)
    used_bytes: list[int] = field(default_factory=list)
    requested_hz: float = 0.0
    read_errors: int = 0

    @property
    def sample_count(self) -> int:
        return len(self.seconds)

    @property
    def duration_s(self) -> float:
        if self.sample_count < 2:
            return 0.0
        return self.seconds[-1] - self.seconds[0]

    @property
    def achieved_hz(self) -> float:
        """What we actually got. A large gap from requested_hz invalidates a run."""
        if self.duration_s <= 0:
            return 0.0
        return (self.sample_count - 1) / self.duration_s


class DeviceSampler:
    """Samples `backend` on a background thread until stopped."""

    def __init__(self, backend: HardwareBackend, hz: float = 10.0) -> None:
        if hz <= 0:
            raise ValueError("sampling rate must be positive")
        self._backend = backend
        self._hz = hz
        self._interval = 1.0 / hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._trace = DeviceTrace(requested_hz=hz)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._trace = DeviceTrace(requested_hz=self._hz)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="kvbench-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> DeviceTrace:
        if self._thread is None:
            raise RuntimeError("sampler was never started")
        self._stop.set()
        self._thread.join(timeout=10.0)
        self._thread = None
        return self._trace

    def wait(self, seconds: float) -> None:
        """Sleep for `seconds`, waking early if the sampler is stopped."""
        self._stop.wait(seconds)

    def _loop(self) -> None:
        origin = time.perf_counter()
        next_at = origin
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                watts = self._backend.power_watts()
                used = self._backend.used_memory_bytes()
            except Exception:
                # A single failed NVML read should not kill a long sweep; we
                # count them so a run with many failures can be discarded.
                self._trace.read_errors += 1
            else:
                self._trace.seconds.append(now - origin)
                self._trace.watts.append(watts)
                self._trace.used_bytes.append(used)

            # Absolute schedule, so read latency does not accumulate into drift.
            next_at += self._interval
            self._stop.wait(max(0.0, next_at - time.perf_counter()))

    def __enter__(self) -> DeviceSampler:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self.stop()
