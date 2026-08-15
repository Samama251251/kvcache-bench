from kvbench.config import ExperimentConfig
from kvbench.estimate import estimate_sweep, format_estimate
from kvbench.results import ResultStore


def make_config(**overrides) -> ExperimentConfig:
    payload = {
        "name": "unit",
        "model": {"id": "meta-llama/Llama-3.1-8B-Instruct"},
        "methods": [
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv", "kind": "eviction"},
        ],
        "benchmarks": [{"suite": "longbench", "tasks": ["qasper"]}],
        "budgets": [0.5, 0.25],
        "generation": {"max_new_tokens": 100},
        "passes": [{"mode": "quality", "samples_per_task": 10, "batch_size": 1}],
        "estimate": {
            "prefill_tokens_per_s": 8000,
            "decode_tokens_per_s": 40,
            "assumed_prompt_tokens": 8000,
            "fixed_overhead_s": 0,
        },
    }
    payload.update(overrides)
    return ExperimentConfig.model_validate(payload)


def test_an_empty_store_gives_a_declared_estimate_and_says_so(tmp_path):
    estimate = estimate_sweep(make_config(), ResultStore(tmp_path))
    # 8000/8000 prefill + 100/40 decode = 3.5 s per sample, 10 samples.
    assert estimate.seconds_per_run == 35.0
    assert estimate.basis == "declared"
    assert "not measurement" in estimate.note


def test_pending_cells_drive_the_total(tmp_path):
    estimate = estimate_sweep(make_config(), ResultStore(tmp_path))
    assert estimate.total_cells == 3  # baseline + snapkv at two budgets
    assert estimate.pending_cells == 3
    assert estimate.total_seconds == 35.0 * 3


def test_completed_cells_are_subtracted(tmp_path, record_factory):
    config = make_config()
    store = ResultStore(tmp_path)
    specs = config.expand()
    store.save(record_factory(specs[0].run_id))

    estimate = estimate_sweep(config, store)
    assert estimate.completed_cells == 1
    assert estimate.pending_cells == 2


def test_the_estimate_switches_to_measured_once_records_exist(tmp_path, record_factory):
    config = make_config()
    store = ResultStore(tmp_path)
    for i, spec in enumerate(config.expand()):
        store.save(
            record_factory(
                spec.run_id,
                spec={**spec.model_dump(mode="json")},
                started_at="2026-08-15T00:00:00+00:00",
                finished_at="2026-08-15T00:02:00+00:00",
            )
        )
        if i >= 2:
            break

    estimate = estimate_sweep(config, store)
    assert estimate.basis == "measured"
    assert estimate.seconds_per_run == 120.0


def test_records_from_another_model_do_not_calibrate_this_one(tmp_path, record_factory):
    config = make_config()
    store = ResultStore(tmp_path)
    for i in range(5):
        store.save(record_factory(f"other-{i}"))  # a different model_id
    assert estimate_sweep(config, store).basis == "declared"


def test_cost_is_only_reported_when_a_price_is_given(tmp_path):
    store = ResultStore(tmp_path)
    assert estimate_sweep(make_config(), store).cost_usd is None

    priced = estimate_sweep(make_config(), store, hourly_usd=0.40)
    assert priced.cost_usd == priced.hours * 0.40
    assert "$" in format_estimate(priced)


def test_an_explicit_price_overrides_the_config(tmp_path):
    config = make_config(estimate={"hourly_usd": 0.40})
    estimate = estimate_sweep(config, ResultStore(tmp_path), hourly_usd=1.50)
    assert estimate.hourly_usd == 1.50


def test_the_formatted_output_names_its_basis(tmp_path):
    text = format_estimate(estimate_sweep(make_config(), ResultStore(tmp_path)))
    assert "declared" in text
    assert "GPU-hours" in text
