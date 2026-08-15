"""Low-bit KV cache, the comparison axis against eviction.

The installed kvpress ships no KIVI-style quantized press, so this goes through
transformers' own quantized cache instead. That is a real difference in
mechanism from the eviction methods and is recorded as such: it shrinks every
token rather than dropping some, which is exactly the contrast the
matched-memory-footprint comparison is meant to expose.
"""

from __future__ import annotations

from contextlib import contextmanager

from ..config import RunSpec
from .base import Method

DEFAULT_BACKEND = "quanto"


class QuantizedCacheMethod(Method):
    kind = "quantization"

    def __init__(self, name: str, spec: RunSpec, params: dict | None = None) -> None:
        super().__init__(name, spec, params)
        if spec.quant_bits is None:
            raise ValueError(f"{name} needs a bit width; none was set on the run spec")
        self.bits = spec.quant_bits
        self.backend = self.params.get("backend", DEFAULT_BACKEND)

    @contextmanager
    def apply(self, model):
        yield model

    def generate_kwargs(self) -> dict:
        return {
            "cache_implementation": "quantized",
            "cache_config": {
                "backend": self.backend,
                "nbits": self.bits,
                **{k: v for k, v in self.params.items() if k != "backend"},
            },
        }

    def describe(self) -> dict:
        info = super().describe()
        info["params"] = {**info["params"], "backend": self.backend}
        return info
