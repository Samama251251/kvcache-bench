from collections import Counter

import pytest
import typer

from kvbench.cli import _parse_shard
from kvbench.config import ExperimentConfig
from kvbench.results import ResultStore
from kvbench.runner import plan_specs


def make_config(**overrides) -> ExperimentConfig:
    payload = {
        "name": "unit",
        "model": {"id": "meta-llama/Llama-3.1-8B-Instruct"},
        "methods": [
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv"},
            {"name": "tova"},
        ],
        "benchmarks": [
            {"suite": "longbench", "tasks": ["qasper", "hotpotqa"]},
            {"suite": "ruler", "tasks": ["niah_single_1"], "context_lengths": [4096, 32768]},
        ],
        "budgets": [0.5, 0.25],
        "generation": {"max_context_tokens": 32768},
    }
    payload.update(overrides)
    return ExperimentConfig.model_validate(payload)


def shards(config, store, count):
    return [plan_specs(config, store, shard=(i, count)) for i in range(count)]


def test_shards_partition_the_sweep_exactly(tmp_path):
    config, store = make_config(), ResultStore(tmp_path)
    whole = plan_specs(config, store)
    parts = shards(config, store, 4)

    ids = [s.run_id for part in parts for s in part]
    assert sorted(ids) == sorted(s.run_id for s in whole)
    assert len(ids) == len(set(ids)), "a cell must not be run by two workers"


def test_shards_are_balanced(tmp_path):
    sizes = [len(part) for part in shards(make_config(), ResultStore(tmp_path), 4)]
    assert max(sizes) - min(sizes) <= 1


def test_every_worker_gets_a_mix_of_methods(tmp_path):
    # This is the property that keeps multi-GPU honest: if one card in the box
    # runs hotter than its neighbours, it must not be the card that ran all of
    # one method, or that method inherits the difference as a result.
    config = ExperimentConfig.from_yaml("configs/full_sweep.yaml")
    for part in shards(config, ResultStore(tmp_path), 4):
        counts = Counter(s.method for s in part)
        assert len(counts) >= 6, f"a worker saw only {sorted(counts)}"
        assert max(counts.values()) / len(part) < 0.35, "one method dominates a worker"


def test_sharding_is_deterministic_across_processes(tmp_path):
    config, store = make_config(), ResultStore(tmp_path)
    first = [s.run_id for s in plan_specs(config, store, shard=(2, 4))]
    second = [s.run_id for s in plan_specs(make_config(), ResultStore(tmp_path), shard=(2, 4))]
    assert first == second


def test_workers_skip_what_any_worker_already_finished(tmp_path, record_factory):
    config, store = make_config(), ResultStore(tmp_path)
    target = plan_specs(config, store, shard=(1, 4))[0]
    store.save(record_factory(target.run_id))

    assert target.run_id not in {s.run_id for s in plan_specs(config, store, shard=(1, 4))}


def test_an_out_of_range_shard_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid shard"):
        plan_specs(make_config(), ResultStore(tmp_path), shard=(4, 4))
    with pytest.raises(ValueError, match="invalid shard"):
        plan_specs(make_config(), ResultStore(tmp_path), shard=(0, 0))


@pytest.mark.parametrize("text,expected", [("0/1", (0, 1)), ("3/4", (3, 4))])
def test_shard_strings_parse(text, expected):
    assert _parse_shard(text) == expected


@pytest.mark.parametrize("text", ["4/4", "-1/2", "half", "1/0", "1"])
def test_bad_shard_strings_are_rejected(text):
    with pytest.raises(typer.BadParameter):
        _parse_shard(text)


# --- method context ceilings --------------------------------------------------


def ceiling_config():
    return make_config(
        methods=[
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv"},
            {"name": "h2o", "max_context_tokens": 16384},
        ]
    )


def test_a_method_is_dropped_above_its_context_ceiling(tmp_path):
    specs = ceiling_config().expand()
    h2o = [s for s in specs if s.method == "h2o"]

    assert h2o, "h2o should still run where it fits"
    assert all(s.context_length != 32768 for s in h2o)
    # LongBench prompts are truncated to 32768, over h2o's ceiling, so h2o is
    # dropped there too rather than being run on different prompts.
    assert all(s.suite == "ruler" for s in h2o)


def test_other_methods_keep_the_cells_the_capped_one_loses(tmp_path):
    specs = ceiling_config().expand()
    assert any(s.method == "snapkv" and s.context_length == 32768 for s in specs)
    assert any(s.method == "snapkv" and s.suite == "longbench" for s in specs)


def test_exclusions_are_reported_with_reasons(tmp_path):
    exclusions = ceiling_config().exclusions()
    names = {name for name, _, _ in exclusions}
    assert names == {"h2o"}
    assert any("16384-token ceiling" in reason for _, _, reason in exclusions)
    assert any("incomparable" in reason for _, _, reason in exclusions)


def test_no_ceiling_means_no_exclusions(tmp_path):
    assert make_config().exclusions() == []
