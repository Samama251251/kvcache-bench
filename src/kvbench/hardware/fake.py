"""A deterministic stand-in for a GPU.

Used on the macOS dev machine and in tests. Power readings follow a fixed
triangular pattern rather than anything random, so a test can assert on exact
joules and a local end-to-end run produces stable numbers.

Nothing this backend reports is a measurement. Records produced with it carry
`backend: "fake"` in their device info, and the analysis layer must never mix
them with real runs.
"""

from __future__ import annotations

from .base import DeviceInfo, HardwareBackend

IDLE_WATTS = 60.0
PEAK_WATTS = 280.0
PERIOD = 20  # samples per full idle -> peak -> idle cycle


class FakeBackend(HardwareBackend):
    def __init__(self, index: int = 0, total_memory_bytes: int = 24 * 1024**3) -> None:
        self._index = index
        self._total_memory = total_memory_bytes
        self._power_calls = 0
        self._memory_calls = 0
        self._locked_mhz: int | None = None

    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            backend="fake",
            index=self._index,
            name="FakeGPU (no measurement)",
            driver_version="0.0.0",
            total_memory_bytes=self._total_memory,
            power_limit_watts=PEAK_WATTS + 20.0,
        )

    def power_watts(self) -> float:
        phase = self._power_calls % PERIOD
        self._power_calls += 1
        # Triangle wave: rises for half the period, falls for the other half.
        rise = phase / (PERIOD / 2) if phase < PERIOD / 2 else (PERIOD - phase) / (PERIOD / 2)
        return IDLE_WATTS + (PEAK_WATTS - IDLE_WATTS) * rise

    def used_memory_bytes(self) -> int:
        # Climbs and plateaus, so peak-memory logic has something to find.
        self._memory_calls += 1
        return int(min(self._memory_calls, 100) * 0.01 * self._total_memory * 0.6)

    def lock_clocks(self, mhz: int) -> bool:
        self._locked_mhz = mhz
        return True

    def reset_clocks(self) -> bool:
        self._locked_mhz = None
        return True
