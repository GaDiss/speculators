"""Load-time fake quantization of cached hidden states (training split only)."""

from os import PathLike
from typing import Any

import torch

from speculators.train.qdq import qdq
from speculators.train.transforms import (
    apply_block_transform,
    hadamard_matrix,
    invert_block_transform,
)

BatchType = dict[str, Any]

DATA_FREE_VARIANT = "hadamard"


class CacheQDQ:
    """Fake-quantize cached hidden states per layer channel.

    Args:
        hidden_size: Verifier hidden size (the per-layer channel width).
        aux_scheme: QDQ scheme for the auxiliary hidden states fed to ``fc``
            (``"fp8" | "int8" | "int4" | "int3" | "int2"``), or ``None`` for bf16.
        teacher_scheme: QDQ scheme for the verifier's final hidden state.
        group: Group size for group-wise integer QDQ and for the transform blocks.
        transform_variant: ``"hadamard"`` (data-free, no bundle needed) or a
            calibrated variant name from the bundle (e.g. ``"wush025"``).
        transform_path: Bundle written by ``scripts/fit_cache_transform.py``,
            ``{variant: {"aux": list of n_aux [n_blocks, g, g], "teacher":
            [n_blocks, g, g]}}``.
    """

    def __init__(
        self,
        hidden_size: int,
        aux_scheme: str | None = None,
        teacher_scheme: str | None = None,
        group: int = 128,
        transform_variant: str | None = None,
        transform_path: str | PathLike | None = None,
    ):
        self.hidden_size = hidden_size
        self.aux_scheme = aux_scheme or None
        self.teacher_scheme = teacher_scheme or None
        self.group = group
        self.transform_variant = transform_variant or None
        if hidden_size % group != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by group {group}"
            )

        self._data_free = (
            self.transform_variant == DATA_FREE_VARIANT and transform_path is None
        )
        if self.transform_variant and transform_path is None and not self._data_free:
            raise ValueError(
                f"transform_variant={self.transform_variant!r} is calibrated and "
                f"needs cache_qdq_transform_path (only {DATA_FREE_VARIANT!r} is "
                "data-free)."
            )
        self._aux_mats, self._aux_inv, self._teacher_mat, self._teacher_inv = (
            self._load_bundle(transform_path, self.transform_variant)
        )
        self._hadamard: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def enabled(self) -> bool:
        return self.aux_scheme is not None or self.teacher_scheme is not None

    def __call__(self, sample: BatchType) -> BatchType:
        if self.aux_scheme is not None:
            aux = sample["hidden_states"].unflatten(1, (-1, self.hidden_size))
            aux = torch.stack(
                [
                    self._qdq_one(
                        aux[:, j],
                        self.aux_scheme,
                        self._aux_mats.get(j),
                        self._aux_inv.get(j),
                    )
                    for j in range(aux.shape[1])
                ],
                dim=1,
            )
            sample["hidden_states"] = aux.flatten(1)
        if self.teacher_scheme is not None:
            sample["verifier_last_hidden_states"] = self._qdq_one(
                sample["verifier_last_hidden_states"],
                self.teacher_scheme,
                self._teacher_mat,
                self._teacher_inv,
            )
        return sample

    def _qdq_one(
        self,
        x: torch.Tensor,
        scheme: str,
        transform: torch.Tensor | None,
        inverse: torch.Tensor | None,
    ) -> torch.Tensor:
        if transform is None and self._data_free:
            transform, inverse = self._hadamard_transform()
        if transform is None or inverse is None:
            return qdq(x, scheme, self.group)
        x = apply_block_transform(x, transform)
        x = qdq(x, scheme, self.group)
        return apply_block_transform(x, inverse)

    def _hadamard_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._hadamard is None:
            n_blocks = self.hidden_size // self.group
            h = hadamard_matrix(self.group).to(torch.bfloat16)
            if not torch.equal(h, h.mT):
                raise AssertionError("hadamard_matrix is no longer symmetric")
            t = h.expand(n_blocks, self.group, self.group).contiguous()
            self._hadamard = (t, t)
        return self._hadamard

    @staticmethod
    def _load_bundle(
        path: str | PathLike | None, variant: str | None
    ) -> tuple[
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if path is None or variant is None:
            return {}, {}, None, None
        bundle = torch.load(path, map_location="cpu")
        if variant not in bundle:
            raise ValueError(
                f"variant {variant!r} not in bundle {path} (has {list(bundle)})"
            )
        orthogonal = variant == DATA_FREE_VARIANT
        # An orthogonal (Hadamard) bundle is perfectly conditioned -> bf16, which
        # keeps it bit-identical to the data-free path. A fitted bundle carries an
        # eigenvalue-power rescaling of unknown conditioning -> fp32.
        dtype = torch.bfloat16 if orthogonal else torch.float32

        def convert(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            t = t.float()
            inv = invert_block_transform(t, orthogonal=orthogonal)
            return t.to(dtype), inv.to(dtype)

        aux_mats: dict[int, torch.Tensor] = {}
        aux_inv: dict[int, torch.Tensor] = {}
        for j, t in enumerate(bundle[variant]["aux"]):
            aux_mats[j], aux_inv[j] = convert(t)
        teacher_mat, teacher_inv = convert(bundle[variant]["teacher"])
        return aux_mats, aux_inv, teacher_mat, teacher_inv
