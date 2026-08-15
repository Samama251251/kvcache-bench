import pytest

from kvbench.hardware.base import DeviceInfo, HardwareBackend
from kvbench.metrics.energy import EnergyMetrics, integrate_joules
from kvbench.metrics.latency import LatencyRecorder, _percentile
from kvbench.metrics.memory import (
    MemoryMetrics,
    driver_peak_bytes,
    dtype_bytes,
    geometry_from_hf_config,
    kv_cache_bytes,
)
from kvbench.metrics.sampler import DeviceSampler, DeviceTrace


class ConstantPowerBackend(HardwareBackend):
    """Analytic source: constant watts, so joules have a closed form."""

    def __init__(self, watts: float = 100.0) -> None:
        self.watts = watts
        self.reads = 0

    def device_info(self) -> DeviceInfo:
        return DeviceInfo("test", 0, "constant", "0", 1024, None)

    def power_watts(self) -> float:
        self.reads += 1
        return self.watts

    def used_memory_bytes(self) -> int:
        return 512


class FlakyBackend(ConstantPowerBackend):
    def power_watts(self) -> float:
        self.reads += 1
        if self.reads % 2 == 0:
            raise OSError("NVML read failed")
        return self.watts


# --- energy integration -------------------------------------------------------
# This is the calculation a silent bug in would corrupt the headline metric, so
# it is checked against signals whose integral is known exactly.


def test_constant_power_integrates_to_power_times_time():
    seconds = [0.0, 1.0, 2.0, 3.0]
    assert integrate_joules(seconds, [50.0] * 4) == pytest.approx(150.0)


def test_linear_ramp_integrates_to_the_triangle_area():
    # 0 W to 100 W over 10 s -> 500 J. Trapezoid is exact for a linear signal.
    seconds = [float(i) for i in range(11)]
    watts = [10.0 * i for i in range(11)]
    assert integrate_joules(seconds, watts) == pytest.approx(500.0)


def test_uneven_sample_spacing_is_handled():
    # 100 W for 1 s then 200 W held to 5 s -> 150 + 800 = ...
    seconds = [0.0, 1.0, 5.0]
    watts = [100.0, 100.0, 200.0]
    assert integrate_joules(seconds, watts) == pytest.approx(100.0 + 600.0)


def test_a_single_sample_yields_zero_not_an_extrapolation():
    assert integrate_joules([1.0], [250.0]) == 0.0
    assert integrate_joules([], []) == 0.0


def test_mismatched_series_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        integrate_joules([0.0, 1.0], [10.0])


def test_time_going_backwards_is_rejected():
    with pytest.raises(ValueError, match="non-decreasing"):
        integrate_joules([0.0, 2.0, 1.0], [10.0, 10.0, 10.0])


def test_energy_metrics_derive_per_token_and_mean_watts():
    trace = DeviceTrace(
        seconds=[0.0, 1.0, 2.0],
        watts=[100.0, 100.0, 100.0],
        used_bytes=[1, 2, 3],
        requested_hz=1.0,
    )
    m = EnergyMetrics.from_trace(trace, generated_tokens=50)
    assert m.total_joules == pytest.approx(200.0)
    assert m.mean_watts == pytest.approx(100.0)
    assert m.peak_watts == pytest.approx(100.0)
    assert m.joules_per_generated_token == pytest.approx(4.0)


def test_joules_per_token_is_null_when_nothing_was_generated():
    trace = DeviceTrace(seconds=[0.0, 1.0], watts=[100.0, 100.0], used_bytes=[1, 1])
    assert EnergyMetrics.from_trace(trace, generated_tokens=0).joules_per_generated_token is None


def test_idle_baseline_is_subtracted_without_touching_the_total():
    trace = DeviceTrace(seconds=[0.0, 2.0], watts=[100.0, 100.0], used_bytes=[1, 1])
    m = EnergyMetrics.from_trace(trace, generated_tokens=10, idle_watts=60.0)
    assert m.total_joules == pytest.approx(200.0)
    assert m.net_joules_above_idle == pytest.approx(80.0)


# --- sampler ------------------------------------------------------------------


def test_sampler_collects_a_trace_and_reports_its_real_rate():
    backend = ConstantPowerBackend(watts=120.0)
    sampler = DeviceSampler(backend, hz=100.0)
    sampler.start()
    sampler.wait(0.3)
    trace = sampler.stop()

    assert trace.sample_count > 5
    assert set(trace.watts) == {120.0}
    assert trace.requested_hz == 100.0
    # Never assert the achieved rate equals the requested one -- it is recorded
    # precisely because a loaded box will not hit it.
    assert 0 < trace.achieved_hz <= 200.0
    assert integrate_joules(trace.seconds, trace.watts) == pytest.approx(120.0 * trace.duration_s)


def test_failed_reads_are_counted_rather_than_killing_the_run():
    sampler = DeviceSampler(FlakyBackend(), hz=100.0)
    sampler.start()
    sampler.wait(0.2)
    trace = sampler.stop()
    assert trace.read_errors > 0
    assert trace.sample_count > 0


def test_sampler_rejects_a_nonpositive_rate():
    with pytest.raises(ValueError, match="positive"):
        DeviceSampler(ConstantPowerBackend(), hz=0)


def test_sampler_cannot_be_started_twice():
    sampler = DeviceSampler(ConstantPowerBackend(), hz=50.0)
    sampler.start()
    with pytest.raises(RuntimeError, match="already started"):
        sampler.start()
    sampler.stop()


def test_stopping_an_unstarted_sampler_is_an_error():
    with pytest.raises(RuntimeError, match="never started"):
        DeviceSampler(ConstantPowerBackend()).stop()


# --- memory -------------------------------------------------------------------


class _Cfg:
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    hidden_size = 4096


def test_kv_cache_size_follows_the_standard_formula():
    # 2 (K and V) * 32 layers * 8 kv heads * 128 head_dim * 1000 tokens * 2 bytes
    got = kv_cache_bytes(num_layers=32, num_kv_heads=8, head_dim=128, tokens=1000, dtype="bfloat16")
    assert got == 2 * 32 * 8 * 128 * 1000 * 2


def test_grouped_query_attention_uses_kv_heads_not_attention_heads():
    geom = geometry_from_hf_config(_Cfg())
    # Llama-3.1-8B shape: 8 KV heads, not 32. Using 32 would overstate every
    # cache size we report by 4x.
    assert geom == {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}


def test_head_dim_falls_back_to_hidden_size_over_heads():
    class NoHeadDim(_Cfg):
        pass

    assert geometry_from_hf_config(NoHeadDim())["head_dim"] == 128


def test_low_bit_dtypes_are_sub_byte():
    assert dtype_bytes("int4") == 0.5
    assert dtype_bytes("int2") == 0.25
    with pytest.raises(ValueError, match="unknown dtype"):
        dtype_bytes("float3")


def test_kv_saving_is_relative_to_the_uncompressed_cache():
    m = MemoryMetrics(
        torch_peak_bytes=None,
        driver_peak_bytes=None,
        kv_cache_bytes=250,
        kv_cache_bytes_uncompressed=1000,
        retained_tokens=250,
        prompt_tokens=1000,
    )
    assert m.kv_cache_saving == pytest.approx(0.75)


def test_driver_peak_is_the_high_water_mark_of_the_trace():
    trace = DeviceTrace(seconds=[0.0, 1.0, 2.0], watts=[1, 1, 1], used_bytes=[10, 90, 40])
    assert driver_peak_bytes(trace) == 90
    assert driver_peak_bytes(DeviceTrace()) is None


# --- latency ------------------------------------------------------------------


def test_ttft_and_itl_are_recorded_separately():
    clock = iter([0.0, 0.5, 0.6, 0.8, 1.0]).__next__
    rec = LatencyRecorder(clock=clock)
    rec.start_sample()  # t=0.0
    rec.mark_token()  # t=0.5 -> TTFT
    rec.mark_token()  # t=0.6 -> ITL 0.1
    rec.mark_token()  # t=0.8 -> ITL 0.2
    rec.end_sample()  # t=1.0

    m = rec.metrics()
    assert m.ttft_s == pytest.approx(0.5)
    assert m.itl_mean_s == pytest.approx(0.15)
    assert m.generated_tokens == 3
    assert m.wall_s == pytest.approx(1.0)
    assert m.throughput_tokens_per_s == pytest.approx(3.0)


def test_metrics_accumulate_across_samples():
    clock = iter([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).__next__
    rec = LatencyRecorder(clock=clock)
    for _ in range(2):
        rec.start_sample()
        rec.mark_token()
        rec.end_sample()
    m = rec.metrics()
    assert m.generated_tokens == 2
    assert m.ttft_s == pytest.approx(1.0)
    # No second token in either sample, so there is no inter-token interval.
    assert m.itl_mean_s is None


def test_marking_a_token_outside_a_sample_is_an_error():
    with pytest.raises(RuntimeError, match="before start_sample"):
        LatencyRecorder().mark_token()


def test_percentiles_return_real_samples():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert _percentile(values, 50) == 3.0
    assert _percentile(values, 95) == 100.0
    assert _percentile([], 50) is None
