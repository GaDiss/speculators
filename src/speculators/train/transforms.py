"""Blockwise data-aware transforms applied before quantization (after WUSH).

Reference: github.com/IST-DASLab/WUSH.

    transform = hadamard @ diag(eig(x.T @ x) ** -exponent) @ eigvec(x.T @ x).T

fit per block of ``group`` channels from the *uncentered* second moment.
"""

import torch

_EIG_EPS = 1e-8


def hadamard_matrix(size: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Sylvester-construction normalized Hadamard matrix, ``size`` a power of 2."""
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError(f"size must be a positive power of 2, got {size}")
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < size:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h / (size**0.5)


def blockwise_gram(x: torch.Tensor, group: int) -> torch.Tensor:
    """Per-block uncentered second moment ``x.T @ x``, blocks of ``group`` channels."""
    if x.shape[-1] % group != 0:
        raise ValueError(f"hidden={x.shape[-1]} not divisible by group={group}")
    xb = x.double().unflatten(-1, (-1, group))  # [n, n_blocks, group]
    return torch.einsum("ngi,ngj->gij", xb, xb)


def fit_block_transform(
    gram: torch.Tensor, exponent: float = 0.25, use_hadamard: bool = True
) -> torch.Tensor:
    """Fit the activation-only transform from a per-block Gram.

    ``exponent`` is the eigenvalue power. ``use_hadamard=False`` gives
    eigenvalue shrinkage with no rotation. Returns ``[n_blocks, group, group]`` fp64,
    applied as ``x_t = x @ transform.mT``.
    """
    n_blocks, group, _ = gram.shape
    hadamard = (
        hadamard_matrix(group, dtype=gram.dtype)
        if use_hadamard
        else torch.eye(group, dtype=gram.dtype)
    )
    if exponent == 0:
        return hadamard.expand(n_blocks, group, group).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)  # ascending, [g,d],[g,d,d]
    eigenvalues = eigenvalues.clamp(min=_EIG_EPS)
    scale = eigenvalues.pow(-exponent)  # [n_blocks, group]
    transform = torch.einsum(
        "is,gs,gjs->gij", hadamard, scale, eigenvectors
    )  # hadamard @ diag(scale) @ eigenvectors.T, per block
    if not transform.isfinite().all():
        raise ValueError("fitted transform contains non-finite values")
    return transform


def invert_block_transform(transform: torch.Tensor, orthogonal: bool) -> torch.Tensor:
    """Inverse of each per-block transform (transpose if orthogonal, else solve)."""
    if orthogonal:
        return transform.mT.contiguous()
    return torch.linalg.inv(transform)


def apply_block_transform(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Per-block transform: ``x_t[..., g, :] = x[..., g, :] @ transform[g].T``."""
    dtype = x.dtype
    compute = torch.promote_types(dtype, transform.dtype)
    xb = x.to(compute).unflatten(-1, (transform.shape[0], transform.shape[1]))
    out = torch.einsum("...gi,gji->...gj", xb, transform.to(compute))
    return out.flatten(-2).to(dtype)
