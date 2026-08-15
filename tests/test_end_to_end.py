"""The whole harness, on this machine, with no GPU.

A real model is loaded and really generates; only the benchmark data and the
power readings are substituted. That is the point of the exercise -- the GPU
session should discover facts about the GPU, not bugs in this code.
"""

from __future__ import annotations

import csv
import json

import pytest

from kvbench.benchmarks.loader import Sample
from kvbench.config import ExperimentConfig
from kvbench.hardware.fake import FakeBackend
from kvbench.results import REQUIRED_AXES, ResultStore
from kvbench.runner import run_sweep

CONFIG = "configs/local_tiny.yaml"


def by_method(records, mode="performance"):
    """Index one pass's records by method name."""
    return {r.method.name: r for r in records if r.spec["mode"] == mode}


@pytest.fixture(scope="module")
def experiment() -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(CONFIG)
    try:
        from transformers import AutoModelForCausalLM

        AutoModelForCausalLM.from_pretrained(config.model.id)
    except Exception as exc:  # noqa: BLE001 - offline or rate-limited, not a failure
        pytest.skip(f"tiny model unavailable ({type(exc).__name__}); needs the HF hub once")
    return config


@pytest.fixture
def offline_samples(monkeypatch):
    """Stand in for LongBench so the harness runs without the hub."""
    samples = [
        Sample(
            context="The capital of France is Paris. " * 40,
            question="\nWhat is the capital of France?",
            answer_prefix="\nAnswer:",
            answers=["Paris"],
            task="qasper",
            extra={"all_classes": None, "length": 200},
        ),
        Sample(
            context="The tallest mountain is Everest. " * 40,
            question="\nWhich mountain is tallest?",
            answer_prefix="\nAnswer:",
            answers=["Everest"],
            task="qasper",
            extra={"all_classes": None, "length": 200},
        ),
    ]
    monkeypatch.setattr(
        "kvbench.runner.loader.load_samples",
        lambda suite, task, **kwargs: samples[: kwargs.get("limit") or len(samples)],
    )
    return samples


def test_a_sweep_runs_end_to_end_and_writes_complete_records(tmp_path, experiment, offline_samples):
    store = ResultStore(tmp_path)
    records = run_sweep(experiment, store, backend=FakeBackend(), resume=True)

    # Baseline plus two presses, one budget, one task -- across both passes.
    assert len(records) == 6
    assert {r.method.name for r in records} == {"full_cache", "streaming_llm", "snapkv"}
    assert {r.spec["mode"] for r in records} == {"quality", "performance"}
    for record in records:
        assert record.status == "ok", record.error

    payload = json.loads(store.path_for(records[0].run_id).read_text())
    for axis in REQUIRED_AXES:
        assert payload[axis], f"{axis} was not populated"


def test_the_baseline_really_is_uncompressed(tmp_path, experiment, offline_samples):
    store = ResultStore(tmp_path)
    records = by_method(run_sweep(experiment, store, backend=FakeBackend()))

    baseline = records["full_cache"]
    # Off by at most one token: the cache is read back after generation, so the
    # boundary between prompt and first generated token is not exact.
    assert baseline.method.effective_compression_ratio == pytest.approx(0.0, abs=0.01)
    assert baseline.memory.kv_cache_bytes == pytest.approx(
        baseline.memory.kv_cache_bytes_uncompressed, rel=0.01
    )


@pytest.mark.parametrize("method", ["snapkv", "streaming_llm"])
def test_presses_actually_evict_to_the_requested_budget(tmp_path, experiment, offline_samples, method):
    store = ResultStore(tmp_path)
    records = by_method(run_sweep(experiment, store, backend=FakeBackend()))

    # The config asks for 50% retention. Measured, not assumed.
    assert records[method].method.effective_compression_ratio == pytest.approx(0.5, abs=0.01)
    assert records[method].memory.retained_tokens < records["full_cache"].memory.retained_tokens


def test_every_run_lands_in_the_index_with_its_hardware_named(tmp_path, experiment, offline_samples):
    store = ResultStore(tmp_path)
    run_sweep(experiment, store, backend=FakeBackend())

    with open(store.index_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert {row["hardware_backend"] for row in rows} == {"fake"}


def test_a_thin_power_trace_is_flagged_on_the_record(tmp_path, experiment, offline_samples):
    store = ResultStore(tmp_path)
    records = run_sweep(experiment, store, backend=FakeBackend())

    # A tiny model on CPU finishes in milliseconds, far too fast to sample a
    # believable power curve. The numbers are still written -- they are a real
    # integral of a real trace -- but the record has to say the trace was thin,
    # so analysis can filter these out instead of plotting them.
    thin = [r for r in records if r.energy.sample_count < 10]
    assert thin, "expected the toy runs to be too short to measure energy well"
    for record in thin:
        assert any("power samples" in w for w in record.warnings)

    with open(store.index_path) as fh:
        rows = list(csv.DictReader(fh))
    assert any(row["warnings"] for row in rows), "warnings must reach the index, not just the JSON"


def test_rerunning_a_finished_sweep_does_nothing(tmp_path, experiment, offline_samples):
    store = ResultStore(tmp_path)
    run_sweep(experiment, store, backend=FakeBackend(), resume=True)
    before = {p: p.stat().st_mtime_ns for p in store.runs_dir.glob("*.json")}

    second = run_sweep(experiment, store, backend=FakeBackend(), resume=True)

    assert second == []
    after = {p: p.stat().st_mtime_ns for p in store.runs_dir.glob("*.json")}
    assert before == after


def test_a_broken_method_fails_its_own_cell_only(tmp_path, experiment, offline_samples, monkeypatch):
    real = ExperimentConfig.from_yaml(CONFIG)

    def explode(config, spec):
        if config.name == "snapkv":
            raise RuntimeError("simulated press failure")
        from kvbench.registry import build_method as original

        return original(config, spec)

    monkeypatch.setattr("kvbench.runner.build_method", explode)
    store = ResultStore(tmp_path)
    records = run_sweep(real, store, backend=FakeBackend())

    failed = [r for r in records if r.status == "failed"]
    assert {r.method.name for r in failed} == {"snapkv"}
    assert all("simulated press failure" in r.error for r in failed)
    assert [r for r in records if r.method.name == "full_cache" and r.status != "ok"] == []

    # Failed cells are offered again on the next resume; the good ones are not.
    done = store.completed_ids()
    assert all(r.run_id not in done for r in failed)
    assert len(done) == len(records) - len(failed)


def test_batched_quality_runs_disclaim_their_own_perf_numbers(tmp_path, experiment, offline_samples):
    store = ResultStore(tmp_path)
    records = run_sweep(experiment, store, backend=FakeBackend())

    quality = [r for r in records if r.spec["mode"] == "quality"]
    assert quality and all(r.spec["batch_size"] > 1 for r in quality)
    for record in quality:
        # Tokens really were generated -- that is a fact about the run. What is
        # not comparable is the per-token timing, and the record has to say so.
        assert record.energy.joules_per_generated_token is not None
        assert any("not comparable" in w for w in record.warnings)
        assert record.latency.ttft_s is None

    performance = [r for r in records if r.spec["mode"] == "performance"]
    for record in performance:
        assert record.latency.ttft_s is not None
        assert not any("not comparable" in w for w in record.warnings)
