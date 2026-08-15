import pytest

from kvbench.config import ExperimentConfig, MethodConfig
from kvbench.methods.full_cache import FullCacheMethod
from kvbench.methods.kvpress_press import KVPressMethod
from kvbench.methods.quantized import QuantizedCacheMethod
from kvbench.registry import PRESS_CLASS_NAMES, build_method, probe, resolve_press_class

SPEC_METHODS = ["streaming_llm", "h2o", "snapkv", "pyramidkv", "tova"]


def specs_for(method: str, kind: str = "eviction", **cfg):
    payload = {
        "name": "unit",
        "model": {"id": "sshleifer/tiny-gpt2", "dtype": "float32"},
        "methods": [
            {"name": "full_cache", "kind": "full_cache"},
            {"name": method, "kind": kind},
        ],
        "benchmarks": [{"suite": "longbench", "tasks": ["qasper"]}],
        "budgets": [0.25],
        "passes": [{"mode": "quality", "samples_per_task": 5}],
    }
    payload.update(cfg)
    config = ExperimentConfig.model_validate(payload)
    return {s.method: s for s in config.expand()}


# --- the Phase 0 gate ---------------------------------------------------------


@pytest.mark.parametrize("method", SPEC_METHODS)
def test_every_spec_method_resolves_in_the_installed_kvpress(method):
    # This is the gate: the spec names these methods, but whether they exist is
    # a property of the installed kvpress, not of the papers.
    status = next(s for s in probe([method]) if s.name == method)
    assert status.available, status.detail


def test_probe_reports_a_missing_press_rather_than_raising(monkeypatch):
    monkeypatch.setitem(PRESS_CLASS_NAMES, "imaginary", "NoSuchPress")
    status = next(s for s in probe(["imaginary"]) if s.name == "imaginary")
    assert not status.available
    assert "NoSuchPress" in status.detail


def test_building_a_missing_press_fails_with_a_pointer_to_doctor(monkeypatch):
    monkeypatch.setitem(PRESS_CLASS_NAMES, "imaginary", "NoSuchPress")
    spec = specs_for("snapkv")["snapkv"]
    with pytest.raises(RuntimeError, match="kvbench doctor"):
        build_method(MethodConfig(name="imaginary"), spec)


def test_an_unknown_method_name_is_rejected():
    spec = specs_for("snapkv")["snapkv"]
    with pytest.raises(KeyError, match="unknown method"):
        build_method(MethodConfig(name="not_a_method"), spec)


# --- adapters -----------------------------------------------------------------


def test_the_baseline_is_a_real_method_not_a_special_case():
    spec = specs_for("snapkv")["full_cache"]
    method = build_method(MethodConfig(name="full_cache", kind="full_cache"), spec)
    assert isinstance(method, FullCacheMethod)
    assert method.generate_kwargs() == {}
    sentinel = object()
    with method.apply(sentinel) as model:
        assert model is sentinel


@pytest.mark.parametrize("method_name", SPEC_METHODS)
def test_one_adapter_covers_every_eviction_method(method_name):
    spec = specs_for(method_name)[method_name]
    method = build_method(MethodConfig(name=method_name), spec)
    assert isinstance(method, KVPressMethod)
    assert method.press_class == PRESS_CLASS_NAMES[method_name]


def test_the_press_is_given_the_discarded_fraction_not_the_retained_one():
    spec = specs_for("snapkv")["snapkv"]  # retention 0.25
    method = build_method(MethodConfig(name="snapkv"), spec)
    assert method.press.compression_ratio == pytest.approx(0.75)
    assert method.describe()["requested_compression_ratio"] == pytest.approx(0.75)


def test_press_params_from_the_config_reach_the_press():
    spec = specs_for("streaming_llm")["streaming_llm"]
    method = build_method(MethodConfig(name="streaming_llm", params={"n_sink": 8}), spec)
    assert method.press.n_sink == 8


def test_quantization_sweeps_bits_and_passes_a_cache_config():
    spec = specs_for("kv_quant", kind="quantization", quant_bits=[4])["kv_quant"]
    method = build_method(MethodConfig(name="kv_quant", kind="quantization"), spec)
    assert isinstance(method, QuantizedCacheMethod)

    kwargs = method.generate_kwargs()
    assert kwargs["cache_implementation"] == "quantized"
    assert kwargs["cache_config"]["nbits"] == 4
    # Quantization shrinks every token instead of dropping some, so it has no
    # retention ratio to report.
    assert method.describe()["requested_compression_ratio"] is None
    assert method.describe()["quant_bits"] == 4


def test_a_quantization_run_without_a_bit_width_is_rejected():
    spec = specs_for("snapkv")["snapkv"]
    with pytest.raises(ValueError, match="bit width"):
        QuantizedCacheMethod("kv_quant", spec)


def test_resolving_an_absent_class_returns_none():
    assert resolve_press_class("DefinitelyNotAPress") is None
