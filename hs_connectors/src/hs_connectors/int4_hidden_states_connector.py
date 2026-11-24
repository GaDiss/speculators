"""Int4-packing sibling of vLLM's ``ExampleHiddenStatesConnector``.

Packs hidden states to group-wise int4 (nibbles + fp16 scales) before writing
them to disk, ~4x smaller than bf16. Only the final tensor-write step is
overridden. Paired on the read side with
:class:`hs_connectors.transfer.Int4Transfer`, which unpacks transparently.
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as _eh_mod,
)

from hs_connectors.int4_utils import INT4_SCALES_KEY, int4_pack


class Int4HiddenStatesConnector(_eh_mod.ExampleHiddenStatesConnector):
    """Packs hidden states to int4 before saving to safetensors.

    The output file contains three tensors::

        hidden_states             - uint8 [seq_len, num_layers, hidden_size // 2]
        hidden_states_int4_scales - fp16  [seq_len, num_layers, hidden_size // 128]
        token_ids                 - int64 [seq_len]
    """

    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        try:
            event.synchronize()
            packed, scales = int4_pack(tensors["hidden_states"])
            save_file(
                {**tensors, "hidden_states": packed, INT4_SCALES_KEY: scales},
                filename,
            )
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
