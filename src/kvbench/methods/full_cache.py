"""The uncompressed baseline.

Not a placeholder: it is a first-class method that runs through exactly the same
loading, warmup, sampling and scoring path as every compressed run. Anything it
did differently would show up as a spurious difference in the numbers everything
else is measured against.
"""

from __future__ import annotations

from contextlib import contextmanager

from .base import Method


class FullCacheMethod(Method):
    kind = "full_cache"

    @contextmanager
    def apply(self, model):
        yield model
