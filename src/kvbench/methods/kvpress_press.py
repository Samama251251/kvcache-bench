"""One adapter for every kvpress eviction method.

kvpress presses share an interface -- constructed with a compression ratio, used
as a context manager over the model -- so StreamingLLM, SnapKV, PyramidKV, TOVA
and observed-attention all go through this single class. There are no per-method
scripts, which is what keeps the comparison honest: no method gets a code path
of its own to be accidentally advantaged by.
"""

from __future__ import annotations

from contextlib import contextmanager

from ..config import RunSpec
from .base import Method


class KVPressMethod(Method):
    kind = "eviction"

    def __init__(self, name: str, spec: RunSpec, press_cls, params: dict | None = None) -> None:
        super().__init__(name, spec, params)
        self._press_cls = press_cls
        self._press = press_cls(compression_ratio=spec.compression_ratio, **self.params)

    @property
    def press_class(self) -> str | None:
        return self._press_cls.__name__

    @property
    def press(self):
        return self._press

    @contextmanager
    def apply(self, model):
        with self._press(model):
            yield model
