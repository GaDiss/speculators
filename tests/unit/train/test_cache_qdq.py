"""Unit tests for the load-time cache QDQ preprocess hook."""

import pytest
import torch

from speculators.train.cache_qdq import CacheQDQ
from speculators.train.qdq import qdq
from speculators.train.transforms import blockwise_gram, fit_block_transform

HIDDEN = 256
N_AUX = 3
GROUP = 128


def _sample(seq_len: int = 40) -> dict:
    torch.manual_seed(0)
    return {
        "hidden_states": torch.randn(seq_len, N_AUX * HIDDEN, dtype=torch.bfloat16),
        "input_ids": torch.arange(seq_len),
        "verifier_last_hidden_states": torch.randn(
            seq_len, HIDDEN, dtype=torch.bfloat16
        ),
        "loss_mask": torch.ones(seq_len, dtype=torch.bool),
    }


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).pow(2).sum().sqrt() / b.float().pow(2).sum().sqrt()
    ).item()


def test_disabled_hook_is_identity():
    hook = CacheQDQ(HIDDEN)
    assert not hook.enabled
    sample = _sample()
    out = hook(dict(sample))
    assert torch.equal(out["hidden_states"], sample["hidden_states"])


def test_qdq_applied_and_bounded():
    base = _sample()
    quant = CacheQDQ(HIDDEN, aux_scheme="int4", teacher_scheme="fp8")(dict(_sample()))

    assert quant["hidden_states"].shape == base["hidden_states"].shape
    assert not torch.equal(quant["hidden_states"], base["hidden_states"])
    assert (
        _relerr(
            quant["verifier_last_hidden_states"], base["verifier_last_hidden_states"]
        )
        < 0.05
    )
    # deterministic: a fixed stored corruption, not augmentation
    again = CacheQDQ(HIDDEN, aux_scheme="int4", teacher_scheme="fp8")(dict(_sample()))
    assert torch.equal(quant["hidden_states"], again["hidden_states"])


def test_aux_channels_quantized_independently():
    """Per-channel QDQ equals quantizing each [seq, hidden] slice on its own."""
    sample = _sample()
    out = CacheQDQ(HIDDEN, aux_scheme="int4")(dict(sample))
    aux = sample["hidden_states"].unflatten(1, (N_AUX, HIDDEN))
    expected = torch.stack([qdq(aux[:, j], "int4") for j in range(N_AUX)], dim=1)
    assert torch.equal(out["hidden_states"], expected.flatten(1))


def test_hadamard_transform_needs_no_bundle(tmp_path):
    """'hadamard' is data-free: a path-free run must equal a fitted-bundle run."""
    gram = torch.eye(GROUP, dtype=torch.float64).expand(HIDDEN // GROUP, GROUP, GROUP)
    had = fit_block_transform(gram, exponent=0.0, use_hadamard=True)
    bundle_path = tmp_path / "bundle.pt"
    torch.save({"hadamard": {"aux": [had] * N_AUX, "teacher": had}}, bundle_path)

    for scheme in ("int4", "int3", "int2"):
        with_bundle = CacheQDQ(
            HIDDEN,
            aux_scheme=scheme,
            teacher_scheme=scheme,
            transform_variant="hadamard",
            transform_path=bundle_path,
        )(dict(_sample()))
        path_free = CacheQDQ(
            HIDDEN,
            aux_scheme=scheme,
            teacher_scheme=scheme,
            transform_variant="hadamard",
        )(dict(_sample()))
        assert torch.equal(with_bundle["hidden_states"], path_free["hidden_states"])
        assert torch.equal(
            with_bundle["verifier_last_hidden_states"],
            path_free["verifier_last_hidden_states"],
        )


def test_hadamard_changes_quantization_but_not_scale():
    """The rotation is applied then inverted: output stays close to the input."""
    base = _sample()
    plain = CacheQDQ(HIDDEN, aux_scheme="int4")(dict(_sample()))
    rotated = CacheQDQ(HIDDEN, aux_scheme="int4", transform_variant="hadamard")(
        dict(_sample())
    )
    assert not torch.equal(plain["hidden_states"], rotated["hidden_states"])
    assert _relerr(rotated["hidden_states"], base["hidden_states"]) < 0.2


def test_calibrated_transform_requires_bundle():
    with pytest.raises(ValueError, match="needs cache_qdq_transform_path"):
        CacheQDQ(HIDDEN, aux_scheme="int4", transform_variant="wush025")


def test_calibrated_bundle_round_trips(tmp_path):
    torch.manual_seed(0)
    mix = torch.randn(HIDDEN, HIDDEN)
    x = torch.randn(2000, HIDDEN) @ mix.mT
    gram = blockwise_gram(x, GROUP)
    t = fit_block_transform(gram, exponent=0.25)
    bundle_path = tmp_path / "bundle.pt"
    torch.save({"wush025": {"aux": [t] * N_AUX, "teacher": t}}, bundle_path)

    hook = CacheQDQ(
        HIDDEN,
        aux_scheme="int8",
        teacher_scheme="int8",
        transform_variant="wush025",
        transform_path=bundle_path,
    )
    base = _sample()
    out = hook(dict(_sample()))
    assert _relerr(out["hidden_states"], base["hidden_states"]) < 0.05


def test_unknown_variant_in_bundle_raises(tmp_path):
    bundle_path = tmp_path / "bundle.pt"
    torch.save({"hadamard": {"aux": [], "teacher": None}}, bundle_path)
    with pytest.raises(ValueError, match="not in bundle"):
        CacheQDQ(
            HIDDEN,
            aux_scheme="int4",
            transform_variant="wush025",
            transform_path=bundle_path,
        )
