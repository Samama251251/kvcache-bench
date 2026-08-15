"""Hardware access behind one interface.

Everything that touches NVML or a physical GPU goes through `HardwareBackend`.
The point is that the sweep orchestration, the power-sampling thread and the
metric maths can all be exercised on a machine with no NVIDIA GPU at all, by
swapping in the fake backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Provenance for the card a run happened on. Copied into every run record."""

    backend: str
    index: int
    name: str
    driver_version: str
    total_memory_bytes: int
    power_limit_watts: float | None

    def as_dict(self) -> dict:
        return asdict(self)


class HardwareBackend(ABC):
    """Instantaneous readings. Sampling and integration live in `kvbench.metrics`."""

    @abstractmethod
    def device_info(self) -> DeviceInfo: ...

    @abstractmethod
    def power_watts(self) -> float:
        """Current board power draw."""

    @abstractmethod
    def used_memory_bytes(self) -> int:
        """Device memory in use, as the driver sees it."""

    def lock_clocks(self, mhz: int) -> bool:
        """Pin the SM clock. Returns whether it was actually applied.

        Unlocked clocks make energy comparisons between methods noisier, because
        the card is free to boost differently from run to run. We record the
        outcome rather than assuming it worked.
        """
        return False

    def reset_clocks(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def __enter__(self) -> HardwareBackend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
