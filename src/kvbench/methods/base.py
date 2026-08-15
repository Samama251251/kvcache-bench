"""The Method interface.

A method is two things to the runner: a context manager to wrap generation in,
and a bag of kwargs to pass to `generate`. Everything else -- what it measures,
how it is scored, where its record lands -- is identical across methods by
construction. That uniformity is the whole point of the harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager

from ..config import RunSpec


class Method(ABC):
    name: str
    kind: str

    def __init__(self, name: str, spec: RunSpec, params: dict | None = None) -> None:
        self.name = name
        self.spec = spec
        self.params = params or {}

    @property
    def press_class(self) -> str | None:
        """The kvpress class backing this method, if any."""
        return None

    @abstractmethod
    @contextmanager
    def apply(self, model):
        """Wrap the model for the duration of one generation."""

    def generate_kwargs(self) -> dict:
        """Extra kwargs for `model.generate`."""
        return {}

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "press_class": self.press_class,
            "requested_compression_ratio": self.spec.compression_ratio,
            "quant_bits": self.spec.quant_bits,
            "params": dict(self.params),
        }
