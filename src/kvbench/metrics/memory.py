"""Memory: the thing compression exists to save, measured two ways plus computed.

Three numbers, deliberately not one:
  - torch's allocator high-water mark (what our process asked for)
  - the driver's used-memory high-water mark (what the card actually held)
  - the KV cache size implied by the model geometry and retained token count

The third is what compression papers quote. Having it next to the two measured
values is what lets us say whether a claimed saving showed up in real VRAM.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .sampler import DeviceTrace

_DTYPE_BYTES = {
    "float32": 4,
    "float": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "int8": 1,
    "int4": 0.5,
    "int2": 0.25,
}


class MemoryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Null where the device cannot report it (CPU/MPS runs, fake backend).
    torch_peak_bytes: int | None
    driver_peak_bytes: int | None
    kv_cache_bytes: int
    kv_cache_bytes_uncompressed: int
    # A KV cache is a per-sequence object, so these describe one representative
    # sequence (the last sample of the run), never a sum over the whole task.
    retained_tokens: int
    prompt_tokens: int

    @property
    def kv_cache_saving(self) -> float:
        if self.kv_cache_bytes_uncompressed == 0:
            return 0.0
        return 1.0 - self.kv_cache_bytes / self.kv_cache_bytes_uncompressed


def dtype_bytes(dtype: str) -> float:
    key = str(dtype).lower().removeprefix("torch.")
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype for KV cache sizing: {dtype}")
    return _DTYPE_BYTES[key]


def kv_cache_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    tokens: int,
    dtype: str,
    batch_size: int = 1,
) -> int:
    """Bytes held by a KV cache of `tokens` positions. Keys and values, hence the 2."""
    per_element = dtype_bytes(dtype)
    return int(2 * batch_size * num_layers * num_kv_heads * head_dim * tokens * per_element)


def geometry_from_hf_config(config) -> dict:
    """Pull the KV-cache-relevant shape out of a transformers config.

    Grouped-query models have fewer KV heads than attention heads, and getting
    that wrong would silently overstate every cache size we report.
    """
    num_attn_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None) or num_attn_heads
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden = getattr(config, "hidden_size", None)
        if hidden is None or not num_attn_heads:
            raise ValueError("cannot derive head_dim from model config")
        head_dim = hidden // num_attn_heads
    return {
        "num_layers": getattr(config, "num_hidden_layers", None) or config.n_layer,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def torch_peak_bytes(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    import torch

    return int(torch.cuda.max_memory_allocated())


def reset_torch_peak(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def driver_peak_bytes(trace: DeviceTrace) -> int | None:
    return max(trace.used_bytes) if trace.used_bytes else None
