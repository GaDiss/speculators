#!/usr/bin/env python
"""Fit blockwise transforms for cache QDQ from a sample of cached hidden states.

CPU-only calibration: accumulates a per-channel blockwise Gram over train-split
samples, then fits the closed-form variants from ``speculators.train.transforms``.
Writes a torch bundle ``{variant: {"aux": [T_0, ..., T_{n-1}], "teacher": T}}``
consumed by ``speculators train --cache-qdq-transform-path`` (channel layout
matches training: all cached layers but the last feed ``fc``, the last is the
teacher).

Usage:
  python scripts/fit_cache_transform.py --data ./output --out ./output/transforms.pt
"""

import argparse
from argparse import Namespace
from pathlib import Path

import torch
from datasets import load_from_disk

from hs_connectors import HiddenStatesBackend
from speculators.train.transforms import blockwise_gram, fit_block_transform

VARIANTS = {
    # name: (eigenvalue exponent, use_hadamard)
    "hadamard": (0.0, True),
    "wush025": (0.25, True),
    "whiten05": (0.5, True),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="prepare-data output directory")
    p.add_argument(
        "--hidden-states-path",
        default=None,
        help="Cache directory (default: <data>/hidden_states)",
    )
    p.add_argument(
        "--hidden-states-backend",
        default="file",
        choices=[n for n in HiddenStatesBackend.registry if n != "mooncake"],
    )
    p.add_argument("--num-calib-samples", type=int, default=200)
    p.add_argument("--group", type=int, default=128)
    p.add_argument(
        "--train-data-ratio",
        type=float,
        default=0.9,
        help="Calibrate on the first N*ratio rows only (the training split).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = load_from_disk(args.data)
    n_train = int(len(ds) * args.train_data_ratio)
    torch.manual_seed(args.seed)
    calib_ids = torch.randperm(n_train)[: args.num_calib_samples].tolist()
    print(f"calibrating on {len(calib_ids)}/{n_train} train-split samples")

    backend = HiddenStatesBackend.registry[args.hidden_states_backend]
    transfer = backend.from_train_args(
        Namespace(hidden_states_path=args.hidden_states_path), args.data
    )

    grams: list[torch.Tensor] = []
    for idx in calib_ids:
        sample = transfer.get_cached(idx)
        if sample is None:
            continue
        hs = sample["hidden_states"]  # [seq, n_layers, hidden]
        if not grams:
            grams = [torch.zeros(0) for _ in range(hs.shape[1])]
        for c in range(hs.shape[1]):
            g = blockwise_gram(hs[:, c], args.group)
            grams[c] = g if grams[c].numel() == 0 else grams[c] + g
    if not grams:
        raise SystemExit("no cached samples found among the calibration indices")

    bundle = {}
    for name, (exponent, use_hadamard) in VARIANTS.items():
        fitted = [
            fit_block_transform(g, exponent=exponent, use_hadamard=use_hadamard)
            for g in grams
        ]
        bundle[name] = {"aux": fitted[:-1], "teacher": fitted[-1]}
        print(f"{name}: fit {len(fitted) - 1} aux channels + teacher")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
