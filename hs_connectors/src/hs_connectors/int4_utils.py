"""Group-wise int4 packing of hidden states (nibbles + fp16 scales, ~4x smaller).

Symmetric absmax quantization over groups of ``group`` channels along the last
dim. Codes are clamped to [-7, 7] (the extra negative code is unused), offset to
[0, 15] and packed two per byte: low nibble = even index, high nibble = odd.

Quantize:   scale = fp16(absmax / 7);  code = clamp(round(x / scale), -7, 7)
Dequantize: x = code * scale
"""

from __future__ import annotations

import torch

from hs_connectors.mooncake_store import assert_finite

INT4_SCALES_KEY = "hidden_states_int4_scales"
INT4_GROUP = 128
_QMAX = 7
_SCALE_EPS = 1e-8


def int4_pack(
    x: torch.Tensor, group: int = INT4_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` over its last dim and pack to nibbles.

    Returns ``(packed uint8 [..., H // 2], scales fp16 [..., H // group])``.
    """
    if x.shape[-1] % group != 0:
        raise ValueError(f"Last dim {x.shape[-1]} not divisible by group {group}")
    assert_finite("hidden_states", x)
    x32 = x.float().unflatten(-1, (-1, group))
    scale = (x32.abs().amax(dim=-1, keepdim=True) / _QMAX).half()
    scale_f = scale.float().clamp(min=_SCALE_EPS)
    codes = (x32 / scale_f).round_().clamp_(-_QMAX, _QMAX).flatten(-2)
    nib = (codes + 8).to(torch.uint8)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()
    return packed, scale.squeeze(-1)


def int4_unpack(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of :func:`int4_pack`. The group size is implied by the shapes."""
    hidden = packed.shape[-1] * 2
    group = hidden // scales.shape[-1]
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    codes = torch.stack([lo, hi], dim=-1).flatten(-2).float()
    scale_f = scales.float().clamp(min=_SCALE_EPS).unsqueeze(-1)
    x = codes.unflatten(-1, (-1, group)) * scale_f
    return x.flatten(-2).to(dtype)
