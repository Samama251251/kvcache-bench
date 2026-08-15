"""The quality/performance split, method context limits, and run ordering."""

import pytest
from pydantic import ValidationError

from kvbench.config import ExperimentConfig, MethodConfig, _batch_for
from kvbench.runner import order_runs


def make_config(**overrides) -> ExperimentConfig:
    base = {
        "name": "unit",
        "model": {"id": "meta-llama/Llama-3.1-8B-Instruct"},
        "methods": [
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv", "kind": "eviction"},
        ],
        "benchmarks": [{"suite": "longbench", "tasks": ["qasper"], "max_context_tokens": 16384}],
        "budgets": [0.5],
        "runtime": {"max_batch_tokens": 65536},
        "passes": [
            {"mode": "quality", "samples_per_task": 50, "batch_size": 8},
            {"mode": "performance", "samples_per_task": 5, "batch_size": 1, "repeats": 3},
        ],
    }
    base.update(overrides)
    return ExperimentConfig.model_validate(base)


# --- passes -------------------------------------------------------------------


def test_both_passes_cover_the_same_grid():
    specs = make_config().expand()
    quality = [s for s in specs if s.mode == "quality"]
    performance = [s for s in specs if s.mode == "performance"]

    assert {s.method for s in quality} == {s.method for s in performance}
    # The performance pass repeats; the quality pass does not.
    assert len(performance) == 3 * len(quality)


def test_quality_and_performance_cells_have_distinct_ids():
    specs = make_config().expand()
    assert len(set(s.run_id for s in specs)) == len(specs)


def test_only_batch_one_runs_count_as_performance_measurements():
    specs = make_config().expand()
    assert all(s.measures_performance for s in specs if s.mode == "performance")
    assert not any(s.measures_performance for s in specs if s.mode == "quality")


def test_a_batched_performance_pass_is_rejected():
    # Batching changes what latency and energy mean; allowing it would let a
    # config quietly produce numbers that look like batch-1 measurements.
    with pytest.raises(ValidationError, match="batch_size 1"):
        make_config(passes=[{"mode": "performance", "samples_per_task": 5, "batch_size": 4}])


def test_a_pass_can_restrict_itself_to_its_own_benchmarks():
    config = make_config(
        passes=[
            {"mode": "quality", "samples_per_task": 50, "batch_size": 8},
            {
                "mode": "performance",
                "samples_per_task": 5,
                "benchmarks": [{"suite": "ruler", "tasks": ["niah_single_1"], "context_lengths": [4096]}],
            },
        ]
    )
    specs = config.expand()
    assert {s.suite for s in specs if s.mode == "quality"} == {"longbench"}
    assert {s.suite for s in specs if s.mode == "performance"} == {"ruler"}


def test_a_sweep_needs_at_least_one_pass():
    with pytest.raises(ValidationError, match="at least one pass"):
        make_config(passes=[])


# --- batch sizing -------------------------------------------------------------


def test_batch_shrinks_as_context_grows():
    # One config has to work at every context length: batch 8 at 8k is the same
    # cache footprint as batch 2 at 32k.
    assert _batch_for(8, 8192, 65536) == 8
    assert _batch_for(8, 16384, 65536) == 4
    assert _batch_for(8, 32768, 65536) == 2


def test_batch_never_drops_below_one():
    assert _batch_for(8, 131072, 65536) == 1


def test_batch_one_stays_one():
    assert _batch_for(1, 4096, 65536) == 1


def test_the_token_budget_reaches_the_specs():
    specs = make_config(
        benchmarks=[{"suite": "ruler", "tasks": ["niah_single_1"], "context_lengths": [32768]}]
    ).expand()
    quality = [s for s in specs if s.mode == "quality"]
    assert {s.batch_size for s in quality} == {2}


# --- method context limits ----------------------------------------------------


def test_a_method_over_its_context_limit_is_dropped_not_attempted():
    config = make_config(
        methods=[
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "h2o", "max_context_tokens": 16384},
        ],
        benchmarks=[{"suite": "ruler", "tasks": ["niah_single_1"], "context_lengths": [4096, 16384, 32768]}],
        passes=[{"mode": "quality", "samples_per_task": 10}],
    )
    h2o = [s for s in config.expand() if s.method == "h2o"]
    baseline = [s for s in config.expand() if s.method == "full_cache"]

    # Eager attention at 32k needs ~69GB for one layer; no single card holds it.
    assert {s.context_length for s in h2o} == {4096, 16384}
    assert {s.context_length for s in baseline} == {4096, 16384, 32768}


def test_dropped_cells_are_reported_rather_than_silent():
    config = make_config(
        methods=[
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "h2o", "max_context_tokens": 16384},
        ],
        benchmarks=[{"suite": "ruler", "tasks": ["niah_single_1"], "context_lengths": [16384, 32768]}],
    )
    assert config.dropped_cells() == [("h2o", 32768)]


def test_a_method_without_a_limit_drops_nothing():
    assert make_config().dropped_cells() == []


# --- run ordering -------------------------------------------------------------


def make_specs(config: ExperimentConfig):
    return config.expand(), {m.name: m for m in config.methods}


def test_ordering_is_deterministic_for_a_seed():
    specs, methods = make_specs(make_config())
    first = [s.run_id for s in order_runs(list(specs), methods, 7)]
    second = [s.run_id for s in order_runs(list(specs), methods, 7)]
    assert first == second


def test_ordering_actually_shuffles():
    specs, methods = make_specs(make_config())
    ordered = order_runs(list(specs), methods, 7)
    assert [s.run_id for s in ordered] != [s.run_id for s in specs]
    assert len(ordered) == len(specs)


def test_eager_attention_runs_are_kept_together():
    config = make_config(
        methods=[
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv", "kind": "eviction"},
            {"name": "h2o", "kind": "eviction"},
        ]
    )
    specs, methods = make_specs(config)
    ordered = order_runs(list(specs), methods, 7)

    # h2o forces eager attention. If its runs were interleaved with the rest the
    # model would reload on almost every cell.
    positions = [i for i, s in enumerate(ordered) if s.method == "h2o"]
    assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_non_eager_methods_are_still_shuffled_among_themselves():
    config = make_config(
        methods=[
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv", "kind": "eviction"},
            {"name": "tova", "kind": "eviction"},
        ]
    )
    specs, methods = make_specs(config)
    ordered = order_runs(list(specs), methods, 3)
    names = [s.method for s in ordered]
    # Not grouped: at least one method reappears after a different one.
    assert len(set(names[:3])) > 1


def test_ordering_survives_an_unknown_method():
    specs, methods = make_specs(make_config())
    assert len(order_runs(list(specs), {}, 1)) == len(specs)


def test_method_config_defaults_to_no_context_limit():
    assert MethodConfig(name="snapkv").max_context_tokens is None
