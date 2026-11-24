"""Unit tests for Int4Transfer (the read side paired with Int4HiddenStatesConnector)."""

import torch
from hs_connectors.int4_utils import INT4_SCALES_KEY, int4_pack, int4_unpack
from hs_connectors.transfer import Int4Transfer
from safetensors.torch import save_file


def _write_packed_sample(path, hidden_states: torch.Tensor, token_ids: torch.Tensor):
    packed, scales = int4_pack(hidden_states)
    save_file(
        {"hidden_states": packed, INT4_SCALES_KEY: scales, "token_ids": token_ids},
        path,
    )


def test_get_cached_unpacks_transparently(tmp_path):
    hidden_states = torch.randn(4, 2, 256, dtype=torch.bfloat16)
    token_ids = torch.arange(4, dtype=torch.long)

    hs_dir = tmp_path / "hidden_states"
    hs_dir.mkdir()
    _write_packed_sample(hs_dir / "hs_7.safetensors", hidden_states, token_ids)

    sample = Int4Transfer(hs_dir).get_cached(7)

    assert sample is not None
    assert INT4_SCALES_KEY not in sample
    assert sample["hidden_states"].dtype == torch.bfloat16
    assert sample["hidden_states"].shape == hidden_states.shape
    assert torch.equal(sample["hidden_states"], int4_unpack(*int4_pack(hidden_states)))
    torch.testing.assert_close(sample["token_ids"], token_ids)


def test_get_generated_unpacks_transparently(tmp_path):
    hidden_states = torch.randn(6, 3, 128, dtype=torch.bfloat16) * 4.0
    token_ids = torch.arange(6, dtype=torch.long)
    handle = tmp_path / "req-1.safetensors"
    _write_packed_sample(handle, hidden_states, token_ids)

    sample = Int4Transfer(tmp_path).get_generated({"hidden_states_path": str(handle)})

    assert sample is not None
    assert INT4_SCALES_KEY not in sample
    assert torch.equal(sample["hidden_states"], int4_unpack(*int4_pack(hidden_states)))


def test_missing_file_returns_none(tmp_path):
    assert Int4Transfer(tmp_path).get_cached(3) is None


def test_passthrough_when_scales_absent(tmp_path):
    hidden_states = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    token_ids = torch.arange(3, dtype=torch.long)
    handle = tmp_path / "req-plain.safetensors"
    save_file({"hidden_states": hidden_states, "token_ids": token_ids}, handle)

    sample = Int4Transfer(tmp_path).get_generated({"hidden_states_path": str(handle)})

    assert sample is not None
    torch.testing.assert_close(sample["hidden_states"], hidden_states)


def test_dequantize_dtype_is_configurable(tmp_path):
    hidden_states = torch.randn(2, 1, 128, dtype=torch.bfloat16)
    handle = tmp_path / "req-fp32.safetensors"
    _write_packed_sample(handle, hidden_states, torch.arange(2))

    transfer = Int4Transfer(tmp_path, dequantize_dtype=torch.float32)
    sample = transfer.get_generated({"hidden_states_path": str(handle)})

    assert sample is not None
    assert sample["hidden_states"].dtype == torch.float32
