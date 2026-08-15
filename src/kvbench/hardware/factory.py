"""Backend selection. Driven by config, never by guessing what card is present."""

from __future__ import annotations

from ..config import RuntimeConfig
from .base import DeviceInfo, HardwareBackend
from .fake import FakeBackend


def make_backend(runtime: RuntimeConfig) -> HardwareBackend:
    if runtime.hardware_backend == "fake":
        return FakeBackend(index=runtime.device_index)
    if runtime.hardware_backend == "nvml":
        from .nvml import NvmlBackend

        return NvmlBackend(index=runtime.device_index)
    raise ValueError(f"unknown hardware backend: {runtime.hardware_backend}")


__all__ = ["make_backend", "HardwareBackend", "DeviceInfo", "FakeBackend"]
