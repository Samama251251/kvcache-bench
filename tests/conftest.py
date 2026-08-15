import pytest

from kvbench.metrics.energy import EnergyMetrics
from kvbench.metrics.latency import LatencyMetrics
from kvbench.metrics.memory import MemoryMetrics
from kvbench.results import Environment, MethodInfo, QualityMetrics, RunRecord


@pytest.fixture
def record_factory():
    """Builds a fully-populated record, so tests can knock out one field at a time."""

    def build(run_id: str = "snapkv-r050-longbench-qasper-rep0-abc123", **overrides) -> RunRecord:
        payload = dict(
            run_id=run_id,
            slug=run_id.rsplit("-", 1)[0],
            spec={
                "experiment": "unit",
                "model_id": "sshleifer/tiny-gpt2",
                "method": "snapkv",
                "mode": "quality",
                "batch_size": 1,
                "retention": 0.5,
                "suite": "longbench",
                "task": "qasper",
                "context_length": None,
                "repeat": 0,
            },
            method=MethodInfo(
                name="snapkv",
                kind="eviction",
                press_class="SnapKVPress",
                requested_compression_ratio=0.5,
                effective_compression_ratio=0.5,
                quant_bits=None,
            ),
            environment=Environment(
                device="cpu",
                device_info={"backend": "fake", "name": "FakeGPU (no measurement)"},
                clocks_locked_mhz=None,
                clock_lock_applied=False,
                torch_version="2.13.0",
                transformers_version="5.2.0",
                kvpress_version="0.5.4",
                platform="test",
                git_sha=None,
                kvbench_version="0.1.0",
            ),
            status="ok",
            started_at="2026-08-15T00:00:00+00:00",
            finished_at="2026-08-15T00:01:00+00:00",
            quality=QualityMetrics(
                primary_metric="qa_f1", primary_score=41.2, scores={"qa_f1": 41.2}, samples_scored=3
            ),
            memory=MemoryMetrics(
                torch_peak_bytes=None,
                driver_peak_bytes=1234,
                kv_cache_bytes=500,
                kv_cache_bytes_uncompressed=1000,
                retained_tokens=500,
                prompt_tokens=1000,
            ),
            latency=LatencyMetrics(
                ttft_s=0.4,
                itl_mean_s=0.02,
                itl_p50_s=0.02,
                itl_p95_s=0.03,
                wall_s=2.0,
                generated_tokens=64,
                throughput_tokens_per_s=32.0,
            ),
            energy=EnergyMetrics(
                total_joules=200.0,
                joules_per_generated_token=3.125,
                mean_watts=100.0,
                peak_watts=140.0,
                idle_watts=None,
                net_joules_above_idle=None,
                duration_s=2.0,
                sample_count=21,
                requested_hz=10.0,
                achieved_hz=10.0,
                read_errors=0,
            ),
        )
        payload.update(overrides)
        return RunRecord(**payload)

    return build
