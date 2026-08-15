import pytest
from pydantic import ValidationError

from kvbench.config import ExperimentConfig


def make_config(**overrides) -> ExperimentConfig:
    base = {
        "name": "unit",
        "model": {"id": "sshleifer/tiny-gpt2", "dtype": "float32"},
        "methods": [
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "snapkv", "kind": "eviction"},
        ],
        "benchmarks": [{"suite": "longbench", "tasks": ["qasper", "hotpotqa"]}],
        "budgets": [0.5, 0.25],
    }
    base.update(overrides)
    return ExperimentConfig.model_validate(base)


def test_expansion_covers_every_cell():
    specs = make_config().expand()
    # baseline (1 budget) + snapkv (2 budgets) = 3 method-budget pairs, 2 tasks each.
    assert len(specs) == 6
    assert {s.method for s in specs} == {"full_cache", "snapkv"}


def test_baseline_is_pinned_to_full_retention():
    specs = [s for s in make_config().expand() if s.method == "full_cache"]
    assert {s.retention for s in specs} == {1.0}
    assert all(s.compression_ratio == 0.0 for s in specs)


def test_compression_ratio_is_the_discarded_fraction():
    # kvpress presses take the fraction thrown away, not the fraction kept.
    spec = next(s for s in make_config().expand() if s.retention == 0.25)
    assert spec.compression_ratio == 0.75


def test_run_ids_are_stable_and_unique():
    first = make_config().expand()
    second = make_config().expand()
    assert [s.run_id for s in first] == [s.run_id for s in second]
    assert len(set(s.run_id for s in first)) == len(first)


def test_run_id_changes_with_the_model():
    a = make_config().expand()[0]
    b = make_config(model={"id": "gpt2", "dtype": "float32"}).expand()[0]
    assert a.run_id != b.run_id


def test_repeats_produce_distinct_runs():
    specs = make_config(repeats=3).expand()
    assert len(specs) == 18
    assert len(set(s.run_id for s in specs)) == 18


def test_ruler_expands_over_context_lengths():
    specs = make_config(
        benchmarks=[{"suite": "ruler", "tasks": ["niah_single_1"], "context_lengths": [4096, 16384]}]
    ).expand()
    assert {s.context_length for s in specs} == {4096, 16384}
    assert "4k" in next(s for s in specs if s.context_length == 4096).slug


def test_quantization_sweeps_bit_widths_not_ratios():
    specs = make_config(
        methods=[
            {"name": "full_cache", "kind": "full_cache"},
            {"name": "kv_quant", "kind": "quantization"},
        ],
        quant_bits=[4, 2],
    ).expand()
    quant = [s for s in specs if s.method == "kv_quant"]
    assert {s.quant_bits for s in quant} == {4, 2}
    assert all(s.retention is None for s in quant)
    assert "4bit" in quant[0].slug or "4bit" in quant[1].slug


def test_sweep_without_a_baseline_is_rejected():
    with pytest.raises(ValidationError, match="full_cache baseline"):
        make_config(methods=[{"name": "snapkv", "kind": "eviction"}])


def test_retention_of_one_is_rejected_as_a_budget():
    with pytest.raises(ValidationError, match="retention ratio"):
        make_config(budgets=[1.0])


def test_ruler_requires_context_lengths():
    with pytest.raises(ValidationError, match="context_lengths"):
        make_config(benchmarks=[{"suite": "ruler", "tasks": ["niah_single_1"]}])


def test_quantization_method_requires_bit_widths():
    with pytest.raises(ValidationError, match="quant_bits"):
        make_config(
            methods=[
                {"name": "full_cache", "kind": "full_cache"},
                {"name": "kv_quant", "kind": "quantization"},
            ]
        )


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValidationError):
        make_config(bugdets=[0.5])
