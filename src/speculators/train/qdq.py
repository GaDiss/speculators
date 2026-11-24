"""Deterministic fake quantize-dequantize (QDQ) for cached hidden states.

Applied at load time so training sees what a physically compressed cache would
return, without regenerating the cache per scheme.
"""

import torch

E4M3_MAX = 448.0
_SCALE_EPS = 1e-8

QDQ_SCHEMES = ("fp8", "int8", "int4", "int3", "int2")
_INT_BITS = {"int8": 8, "int4": 4, "int3": 3, "int2": 2}


def bytes_per_token_channel(
    scheme: str | None, hidden_size: int, group: int = 128
) -> int:
    """Stored bytes per (token, layer) incl. fp16 scales. ``None`` = bf16.

    fp8: one fp16 scale per (token, layer). int schemes: one fp16 scale per
    ``group`` channels, codes bit-packed.
    """
    if scheme is None:
        return 2 * hidden_size
    if scheme == "fp8":
        return hidden_size + 2
    if scheme in _INT_BITS:
        packed_bytes = -(-hidden_size * _INT_BITS[scheme] // 8)
        return packed_bytes + 2 * (hidden_size // group)
    raise ValueError(f"Unknown QDQ scheme: {scheme!r}")


def fp8_token_qdq(x: torch.Tensor) -> torch.Tensor:
    """Absmax-scaled fp8-E4M3 QDQ with one fp16 scale per slice of the last dim."""
    x32 = x.float()
    scale = (x32.abs().amax(dim=-1, keepdim=True) / E4M3_MAX).half().float()
    scale = scale.clamp(min=_SCALE_EPS)
    q = (x32 / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(torch.bfloat16)


def int_group_qdq(x: torch.Tensor, bits: int = 4, group: int = 128) -> torch.Tensor:
    """Symmetric absmax group-wise integer QDQ over the last dim."""
    if x.shape[-1] % group != 0:
        raise ValueError(
            f"Last dim {x.shape[-1]} is not divisible by group size {group}"
        )
    qmax = 2 ** (bits - 1) - 1
    x32 = x.float().unflatten(-1, (-1, group))
    scale = (x32.abs().amax(dim=-1, keepdim=True) / qmax).half().float()
    scale = scale.clamp(min=_SCALE_EPS)
    q = (x32 / scale).round_().clamp_(-qmax, qmax)
    return (q * scale).flatten(-2).to(torch.bfloat16)


def qdq(x: torch.Tensor, scheme: str, group: int = 128) -> torch.Tensor:
    """Apply a named QDQ scheme to ``x`` (any shape; quantized over the last dim)."""
    if scheme == "fp8":
        return fp8_token_qdq(x)
    if scheme in _INT_BITS:
        return int_group_qdq(x, bits=_INT_BITS[scheme], group=group)
    raise ValueError(f"Unknown QDQ scheme: {scheme!r}")
