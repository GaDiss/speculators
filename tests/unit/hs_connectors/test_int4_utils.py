"""Unit tests for group-wise int4 packing of hidden states."""

import pytest
import torch
from hs_connectors.int4_utils import INT4_SCALES_KEY, int4_pack, int4_unpack
from hs_connectors.mooncake_store import NonFiniteTensorError


@pytest.mark.parametrize("shape", [(37, 4096), (12, 6, 4096), (5, 256)])
def test_pack_shapes_and_dtypes(shape):
    group = shape[-1] if shape[-1] < 128 else 128
    x = torch.randn(*shape, dtype=torch.bfloat16) * 4.0
    packed, scales = int4_pack(x, group=group)
    assert packed.dtype == torch.uint8
    assert packed.shape == (*shape[:-1], shape[-1] // 2)
    assert scales.dtype == torch.float16
    assert scales.shape == (*shape[:-1], shape[-1] // group)


def test_round_trip_is_close():
    torch.manual_seed(0)
    x = torch.randn(64, 3, 256, dtype=torch.bfloat16)
    un = int4_unpack(*int4_pack(x))
    assert un.dtype == torch.bfloat16
    assert un.shape == x.shape
    relerr = (un.float() - x.float()).norm() / x.float().norm()
    assert relerr < 0.2


def test_codes_stay_in_symmetric_range():
    x = torch.randn(8, 128, dtype=torch.bfloat16) * 10.0
    packed, _ = int4_pack(x)
    lo = (packed & 0x0F).int() - 8
    hi = ((packed >> 4) & 0x0F).int() - 8
    assert lo.min() >= -7
    assert lo.max() <= 7
    assert hi.min() >= -7
    assert hi.max() <= 7


def test_group_isolation():
    """An outlier in one group must not change neighbouring groups' output."""
    torch.manual_seed(0)
    x = torch.randn(1, 4096, dtype=torch.bfloat16)
    x_out = x.clone()
    x_out[0, 5] = 22528.0
    dq = int4_unpack(*int4_pack(x))
    dq_out = int4_unpack(*int4_pack(x_out))
    assert torch.equal(dq[0, 128:], dq_out[0, 128:])
    assert not torch.equal(dq[0, :128], dq_out[0, :128])


def test_all_zero_tensor_is_finite_and_exact():
    x = torch.zeros(4, 256, dtype=torch.bfloat16)
    un = int4_unpack(*int4_pack(x))
    assert torch.equal(un, x)


def test_size_ratio():
    """Packed nibbles + fp16 scales ~= 0.258x the bf16 byte count (g128)."""
    x = torch.randn(1000, 6, 4096, dtype=torch.bfloat16)
    packed, scales = int4_pack(x)
    ratio = (packed.numel() + scales.numel() * 2) / (x.numel() * 2)
    assert abs(ratio - 0.2578) < 0.001


def test_requires_divisible_group():
    with pytest.raises(ValueError, match="not divisible"):
        int4_pack(torch.randn(3, 100), group=128)


def test_rejects_non_finite():
    x = torch.randn(4, 128)
    x[0, 0] = float("inf")
    with pytest.raises(NonFiniteTensorError):
        int4_pack(x)


def test_unpack_dtype_is_configurable():
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    assert int4_unpack(*int4_pack(x), dtype=torch.float32).dtype == torch.float32


def test_scales_key_constant():
    assert INT4_SCALES_KEY == "hidden_states_int4_scales"
