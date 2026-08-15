import csv
import json

import pytest
from pydantic import ValidationError

from kvbench.config import ExperimentConfig
from kvbench.results import (
    INDEX_COLUMNS,
    REQUIRED_AXES,
    ResultStore,
    RunRecord,
    audit_completeness,
)


def test_a_record_missing_any_axis_is_rejected(record_factory):
    full = record_factory().model_dump(mode="json")
    for axis in REQUIRED_AXES:
        partial = {k: v for k, v in full.items() if k != axis}
        with pytest.raises(ValidationError):
            RunRecord.model_validate(partial)


def test_a_complete_record_audits_clean(record_factory):
    assert audit_completeness(record_factory()) == []


def test_the_index_row_carries_every_headline_number(record_factory):
    row = record_factory().flat_row()
    assert set(row) == set(INDEX_COLUMNS)
    for key in (
        "primary_score",
        "driver_peak_bytes",
        "kv_cache_bytes",
        "ttft_s",
        "itl_mean_s",
        "throughput_tokens_per_s",
        "total_joules",
        "joules_per_generated_token",
        "peak_watts",
    ):
        assert row[key] is not None, f"{key} missing from the index row"


def test_the_index_row_names_the_hardware(record_factory):
    # Energy and latency are not comparable across cards, so every row has to
    # say which one it came from.
    row = record_factory().flat_row()
    assert row["gpu"] == "FakeGPU (no measurement)"
    assert row["hardware_backend"] == "fake"


def test_save_writes_the_record_and_refreshes_the_index(tmp_path, record_factory):
    store = ResultStore(tmp_path)
    record = record_factory()
    path = store.save(record)

    assert json.loads(path.read_text())["run_id"] == record.run_id
    with open(store.index_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["method"] == "snapkv"


def test_save_leaves_no_temporary_files_behind(tmp_path, record_factory):
    store = ResultStore(tmp_path)
    store.save(record_factory())
    assert list(tmp_path.rglob("*.tmp")) == []


def test_the_index_is_rebuilt_from_the_run_files(tmp_path, record_factory):
    store = ResultStore(tmp_path)
    store.save(record_factory("run-a"))
    store.save(record_factory("run-b"))
    store.index_path.unlink()
    store.rebuild_index()

    with open(store.index_path) as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_completed_ids_only_counts_successful_runs(tmp_path, record_factory):
    store = ResultStore(tmp_path)
    store.save(record_factory("run-ok"))
    store.save(record_factory("run-bad", status="failed", error="CUDA OOM"))
    assert store.completed_ids() == {"run-ok"}


def test_a_corrupt_run_file_does_not_break_resume(tmp_path, record_factory):
    store = ResultStore(tmp_path)
    store.save(record_factory("run-ok"))
    (store.runs_dir / "truncated.json").write_text("{not json")
    assert store.completed_ids() == {"run-ok"}


def test_pending_skips_finished_cells_and_keeps_failed_ones(tmp_path, record_factory):
    config = ExperimentConfig.model_validate(
        {
            "name": "unit",
            "model": {"id": "sshleifer/tiny-gpt2", "dtype": "float32"},
            "methods": [
                {"name": "full_cache", "kind": "full_cache"},
                {"name": "snapkv", "kind": "eviction"},
            ],
            "benchmarks": [{"suite": "longbench", "tasks": ["qasper"]}],
            "budgets": [0.5],
            "passes": [{"mode": "quality", "samples_per_task": 5}],
        }
    )
    specs = config.expand()
    store = ResultStore(tmp_path)
    assert len(store.pending(specs)) == len(specs)

    store.save(record_factory(specs[0].run_id))
    store.save(record_factory(specs[1].run_id, status="failed", error="boom"))

    pending = store.pending(specs)
    # The successful cell is skipped; the failed one is offered again.
    assert specs[0].run_id not in {s.run_id for s in pending}
    assert specs[1].run_id in {s.run_id for s in pending}


def test_records_round_trip_through_disk(tmp_path, record_factory):
    store = ResultStore(tmp_path)
    original = record_factory()
    store.save(original)
    assert store.load(original.run_id) == original


def test_an_empty_store_reports_nothing_done(tmp_path):
    store = ResultStore(tmp_path)
    assert store.completed_ids() == set()
    assert store.all_records() == []
