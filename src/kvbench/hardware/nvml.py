"""NVML-backed hardware access.

Written on a machine with no NVIDIA GPU and first executed during the smoke
session. `nvidia-ml-py` is an optional dependency (the `gpu` extra), so the
import is deliberately deferred to construction time -- importing this module
must stay safe everywhere.
"""

from __future__ import annotations

import subprocess

from .base import DeviceInfo, HardwareBackend


class NvmlBackend(HardwareBackend):
    def __init__(self, index: int = 0) -> None:
        try:
            import pynvml
        except ImportError as exc:  # pragma: no cover - depends on host
            raise RuntimeError("NVML backend requires the 'gpu' extra: uv sync --extra gpu") from exc

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._index = index
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
        self._locked_mhz: int | None = None

    def device_info(self) -> DeviceInfo:
        name = self._nvml.nvmlDeviceGetName(self._handle)
        if isinstance(name, bytes):
            name = name.decode()
        driver = self._nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        try:
            limit = self._nvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
        except self._nvml.NVMLError:
            limit = None
        return DeviceInfo(
            backend="nvml",
            index=self._index,
            name=name,
            driver_version=driver,
            total_memory_bytes=int(mem.total),
            power_limit_watts=limit,
        )

    def power_watts(self) -> float:
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def used_memory_bytes(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def lock_clocks(self, mhz: int) -> bool:
        # nvmlDeviceSetGpuLockedClocks needs root on most hosts, so shell out to
        # nvidia-smi and report honestly whether it took.
        ok = self._nvidia_smi(["-i", str(self._index), "-lgc", f"{mhz},{mhz}"])
        self._locked_mhz = mhz if ok else None
        return ok

    def reset_clocks(self) -> bool:
        ok = self._nvidia_smi(["-i", str(self._index), "-rgc"])
        if ok:
            self._locked_mhz = None
        return ok

    @staticmethod
    def _nvidia_smi(args: list[str]) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", *args],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return True

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # pragma: no cover - shutdown is best effort
            pass
