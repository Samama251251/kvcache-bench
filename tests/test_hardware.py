import pytest

from kvbench.config import RuntimeConfig
from kvbench.hardware import make_backend
from kvbench.hardware.fake import IDLE_WATTS, PEAK_WATTS, FakeBackend


def test_factory_honours_the_configured_backend():
    backend = make_backend(RuntimeConfig(device="cpu", hardware_backend="fake"))
    assert isinstance(backend, FakeBackend)


def test_factory_rejects_unknown_backends():
    runtime = RuntimeConfig.model_construct(hardware_backend="guess")
    with pytest.raises(ValueError, match="unknown hardware backend"):
        make_backend(runtime)


def test_fake_readings_are_reproducible():
    a = [FakeBackend().power_watts() for _ in range(50)]
    b = [FakeBackend().power_watts() for _ in range(50)]
    assert a == b


def test_fake_power_stays_within_its_declared_envelope():
    readings = [FakeBackend().power_watts() for _ in range(200)]
    assert min(readings) >= IDLE_WATTS
    assert max(readings) <= PEAK_WATTS


def test_fake_device_info_is_labelled_as_not_a_measurement():
    info = FakeBackend().device_info()
    # Analysis must be able to filter synthetic runs out on this field alone.
    assert info.backend == "fake"
    assert "fake" in info.name.lower()
    assert info.as_dict()["backend"] == "fake"


def test_fake_memory_is_monotonic():
    backend = FakeBackend()
    readings = [backend.used_memory_bytes() for _ in range(30)]
    assert readings == sorted(readings)
    assert readings[-1] <= backend.device_info().total_memory_bytes


def test_clock_locking_reports_whether_it_applied():
    backend = FakeBackend()
    assert backend.lock_clocks(1500) is True
    assert backend.reset_clocks() is True
