"""
Helper that walks a model and replaces matching ``nn.Linear`` modules with
QuEST's ``QuantizedLinear``, copying weights and biases.
"""

import re
from typing import TYPE_CHECKING

import torch.nn as nn

from speculators.quantization.config import QuantizationConfig
from speculators.quantization.base_linear import QuantizedLinear

__all__ = ["apply_quantization"]


def apply_quantization(
    model: nn.Module,
    config: "QuantizationConfig | None",
) -> int:
    """Replace matching ``nn.Linear`` submodules with ``QuantizedLinear`` in place.

    Args:
        model: Module to mutate. Module paths are resolved via
            ``model.named_modules()``.
        config: Quantization configuration. If ``None``, no-op.

    Returns:
        Number of modules replaced.
    """
    if config is None:
        return 0

    targets = [re.compile(p) for p in config.targets]

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, QuantizedLinear):
            continue
        #  regex
        if not any(p.search(name) for p in targets):
            continue

        new = QuantizedLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            weight_quantizer=config.weight.build(),
            activation_quantizer=config.activation.build(),
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        new.weight = module.weight
        if module.bias is not None:
            new.bias = module.bias

        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, new)
        replaced += 1

    return replaced
