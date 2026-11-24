"""Unit tests for the blockwise (WUSH-style) transforms used before QDQ."""

import pytest
import torch

from speculators.train.transforms import (
    apply_block_transform,
    blockwise_gram,
    fit_block_transform,
    hadamard_matrix,
    invert_block_transform,
)


def test_hadamard_orthogonal():
    h = hadamard_matrix(128)
    assert torch.allclose(h @ h.mT, torch.eye(128, dtype=h.dtype), atol=1e-10)


def test_hadamard_requires_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        hadamard_matrix(100)


def test_blockwise_gram_shape_and_symmetry():
    x = torch.randn(50, 512)
    gram = blockwise_gram(x, group=128)
    assert gram.shape == (4, 128, 128)
    assert torch.allclose(gram, gram.mT)  # x.T @ x is always symmetric
    # positive semi-definite: all eigenvalues >= 0
    eigenvalues = torch.linalg.eigvalsh(gram)
    assert (eigenvalues >= -1e-6).all()


def test_blockwise_gram_requires_divisible_group():
    x = torch.randn(10, 100)
    with pytest.raises(ValueError, match="divisible"):
        blockwise_gram(x, group=128)


@pytest.mark.parametrize(
    ("exponent", "use_hadamard", "orthogonal"),
    [(0.0, True, True), (0.25, True, False), (0.5, True, False), (0.5, False, False)],
)
def test_transform_roundtrip(exponent, use_hadamard, orthogonal):
    torch.manual_seed(0)
    mix = torch.randn(512, 512)
    x = torch.randn(500, 512) @ mix.mT  # correlated within blocks

    gram = blockwise_gram(x, group=128)
    transform = fit_block_transform(gram, exponent=exponent, use_hadamard=use_hadamard)
    inv = invert_block_transform(transform, orthogonal=orthogonal)

    x_t = apply_block_transform(x, transform)
    x_rt = apply_block_transform(x_t, inv)
    assert torch.allclose(x_rt, x, atol=1e-4, rtol=1e-4)


def test_hadamard_only_is_data_independent():
    """exponent=0 must skip the eigendecomposition entirely: same matrix regardless
    of the calibration data, and identical across all blocks."""
    gram_a = blockwise_gram(torch.randn(50, 256), group=128)
    gram_b = blockwise_gram(torch.randn(50, 256) * 100.0, group=128)
    t_a = fit_block_transform(gram_a, exponent=0.0, use_hadamard=True)
    t_b = fit_block_transform(gram_b, exponent=0.0, use_hadamard=True)
    assert torch.equal(t_a, t_b)
    assert torch.equal(t_a[0], t_a[1])  # shared across blocks


def test_transform_suppresses_outliers():
    """A block with one dominant, high-variance direction should see its max
    magnitude shrink after the transform (the whole point of the exercise)."""
    torch.manual_seed(0)
    mix = torch.eye(128)
    mix[0, 0] = 50.0  # one channel with far larger variance than the rest
    x = torch.randn(2000, 128) @ mix.mT

    gram = blockwise_gram(x, group=128)
    transform = fit_block_transform(gram, exponent=0.5, use_hadamard=True)
    x_t = apply_block_transform(x, transform)
    assert x_t.abs().max() < x.abs().max()


def test_degenerate_zero_gram_stays_finite():
    """An all-zero calibration block (dead channel group) must not produce NaN/Inf."""
    gram = torch.zeros(2, 64, 64)
    transform = fit_block_transform(gram, exponent=0.25, use_hadamard=True)
    assert torch.isfinite(transform).all()


def test_apply_block_transform_shape_dtype_preserved():
    x = torch.randn(37, 256, dtype=torch.bfloat16)
    gram = blockwise_gram(x, group=128)
    transform = fit_block_transform(gram, exponent=0.25)
    y = apply_block_transform(x, transform)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16
