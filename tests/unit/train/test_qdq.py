"""Unit tests for quantize-dequantize (QDQ) hidden-state compression simulation."""

import pytest
import torch
from hs_connectors.int4_utils import int4_pack, int4_unpack

from speculators.train.qdq import (
    bytes_per_token_channel,
    fp8_token_qdq,
    int_group_qdq,
    qdq,
)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).pow(2).sum().sqrt() / b.float().pow(2).sum().sqrt()
    ).item()


@pytest.mark.parametrize("scheme", ["fp8", "int4", "int8", "int3", "int2"])
def test_qdq_shape_dtype_preserved(scheme):
    x = torch.randn(37, 4096, dtype=torch.bfloat16) * 4.0
    y = qdq(x, scheme)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16


@pytest.mark.parametrize("scheme", ["fp8", "int4", "int8", "int3", "int2"])
def test_qdq_deterministic(scheme):
    x = torch.randn(16, 4096, dtype=torch.bfloat16) * 3.0
    assert torch.equal(qdq(x, scheme), qdq(x, scheme))


def test_qdq_error_ordering():
    x = torch.randn(64, 4096, dtype=torch.bfloat16)
    e_int8 = _relerr(int_group_qdq(x, bits=8), x)
    e_fp8 = _relerr(fp8_token_qdq(x), x)
    e_int4 = _relerr(int_group_qdq(x, bits=4), x)
    assert e_int8 < e_fp8 < e_int4
    assert e_int4 < 0.2


def test_qdq_error_monotonic_with_bits():
    x = torch.randn(64, 4096, dtype=torch.bfloat16)
    errs = [_relerr(int_group_qdq(x, bits=b), x) for b in (8, 4, 3, 2)]
    assert errs == sorted(errs)


def test_int4_group_isolation():
    """An outlier in one group must not perturb neighbouring groups."""
    torch.manual_seed(0)
    x = torch.randn(1, 4096, dtype=torch.bfloat16)
    x_out = x.clone()
    x_out[0, 5] = 22528.0
    dq = int_group_qdq(x, bits=4, group=128)
    dq_out = int_group_qdq(x_out, bits=4, group=128)
    assert torch.equal(dq[0, 128:], dq_out[0, 128:])
    assert not torch.equal(dq[0, :128], dq_out[0, :128])


def test_int4_symmetric_code_range():
    x = torch.randn(8, 128, dtype=torch.bfloat16) * 10.0
    dq = int_group_qdq(x, bits=4, group=128)
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7
    scale = scale.half().float()
    codes = (dq.float() / scale).round()
    assert codes.abs().max() <= 7


def test_qdq_zero_input_safe():
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    for scheme in ["fp8", "int4", "int8", "int3", "int2"]:
        y = qdq(x, scheme)
        assert torch.isfinite(y.float()).all()
        assert torch.equal(y, x)


def test_int_group_requires_divisible_group():
    x = torch.randn(3, 100, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible"):
        int_group_qdq(x, bits=4, group=128)


def test_fp8_scale_is_per_slice():
    x = torch.randn(3, 4096, dtype=torch.bfloat16)
    x[1] *= 1000.0
    y = qdq(x, "fp8")
    for r in range(3):
        assert _relerr(y[r], x[r]) < 0.05


def test_unknown_scheme_raises():
    x = torch.randn(2, 4096, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Unknown QDQ scheme"):
        qdq(x, "fp4")


@pytest.mark.parametrize("shape", [(37, 4096), (12, 6, 4096), (5, 256)])
def test_int4_qdq_matches_physical_pack(shape):
    """The int4 fake-quant must be bit-identical to the int4 storage backend."""
    group = shape[-1] if shape[-1] < 128 else 128
    x = torch.randn(*shape, dtype=torch.bfloat16) * 4.0
    assert torch.equal(
        int4_unpack(*int4_pack(x, group=group)), int_group_qdq(x, bits=4, group=group)
    )


def test_int4_reqdq_idempotent():
    x = torch.randn(20, 4096, dtype=torch.bfloat16) * 3.0
    un = int4_unpack(*int4_pack(x))
    assert torch.equal(qdq(un, "int4"), un)


def test_bytes_per_token_channel():
    h = 4096
    assert bytes_per_token_channel(None, h) == 8192
    assert bytes_per_token_channel("fp8", h) == 4098
    assert bytes_per_token_channel("int4", h) == 2112
    assert bytes_per_token_channel("int8", h) == 4160
    assert bytes_per_token_channel("int3", h) == 1600
    assert bytes_per_token_channel("int2", h) == 1088
    assert bytes_per_token_channel("int4", h, group=64) == 2048 + 2 * 64
    with pytest.raises(ValueError, match="Unknown QDQ scheme"):
        bytes_per_token_channel("fp4", h)
