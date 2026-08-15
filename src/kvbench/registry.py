"""Method registry, and the Phase 0 gate as code.

The spec names six methods. Which of them actually exist is a property of the
*installed* kvpress, not of the paper they came from, so nothing here assumes a
class is present. `probe()` reports what resolved and what did not, and `kvbench
doctor` prints it. A method that cannot be resolved is dropped explicitly and
shows up in the report as dropped -- four methods measured rigorously beat six
measured sloppily.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MethodConfig, RunSpec
from .methods.base import Method
from .methods.full_cache import FullCacheMethod
from .methods.kvpress_press import KVPressMethod
from .methods.quantized import QuantizedCacheMethod

# Our method name -> the kvpress class we expect to back it. Names on the left
# are the ones that appear in configs, plots and the report.
PRESS_CLASS_NAMES: dict[str, str] = {
    "streaming_llm": "StreamingLLMPress",
    "h2o": "ObservedAttentionPress",
    "snapkv": "SnapKVPress",
    "pyramidkv": "PyramidKVPress",
    "tova": "TOVAPress",
    # Cheap reference points: a scoring-free norm heuristic and a random
    # baseline. If a method cannot beat random eviction at the same budget,
    # that is worth knowing.
    "knorm": "KnormPress",
    "expected_attention": "ExpectedAttentionPress",
    "random": "RandomPress",
}

BUILTIN_KINDS = {"full_cache": FullCacheMethod, "quantization": QuantizedCacheMethod}


@dataclass(frozen=True)
class MethodStatus:
    name: str
    kind: str
    press_class: str | None
    available: bool
    detail: str


def resolve_press_class(press_class_name: str):
    """Look a press class up in the installed kvpress. Returns None if absent."""
    try:
        import kvpress
    except ImportError:
        return None
    return getattr(kvpress, press_class_name, None)


def probe(method_names: list[str] | None = None) -> list[MethodStatus]:
    """Report which methods this install can actually run."""
    names = method_names or ["full_cache", *PRESS_CLASS_NAMES, "kv_quant"]
    statuses = []
    for name in names:
        statuses.append(_probe_one(name))
    return statuses


def _probe_one(name: str) -> MethodStatus:
    if name == "full_cache":
        return MethodStatus(name, "full_cache", None, True, "built-in baseline")
    if name not in PRESS_CLASS_NAMES:
        # Anything not backed by a press is treated as the quantization axis.
        available, detail = _probe_quantized_cache()
        return MethodStatus(name, "quantization", None, available, detail)

    class_name = PRESS_CLASS_NAMES[name]
    cls = resolve_press_class(class_name)
    if cls is None:
        return MethodStatus(name, "eviction", class_name, False, f"kvpress has no {class_name}")
    return MethodStatus(name, "eviction", class_name, True, f"kvpress.{class_name}")


def _probe_quantized_cache() -> tuple[bool, str]:
    try:
        from transformers import QuantizedCacheConfig  # noqa: F401
    except ImportError:
        pass
    else:
        return True, "transformers QuantizedCache"
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, "transformers is not installed"
    return True, "transformers cache_implementation='quantized'"


def build_method(config: MethodConfig, spec: RunSpec) -> Method:
    """Construct the method for one run, or fail loudly."""
    if config.kind in BUILTIN_KINDS:
        return BUILTIN_KINDS[config.kind](config.name, spec, config.params)

    class_name = PRESS_CLASS_NAMES.get(config.name)
    if class_name is None:
        raise KeyError(
            f"unknown method '{config.name}'. Known presses: {sorted(PRESS_CLASS_NAMES)}"
        )
    cls = resolve_press_class(class_name)
    if cls is None:
        raise RuntimeError(
            f"method '{config.name}' needs kvpress.{class_name}, which the installed "
            f"kvpress does not provide. Run 'kvbench doctor' and drop it from the config."
        )
    return KVPressMethod(config.name, spec, cls, config.params)
