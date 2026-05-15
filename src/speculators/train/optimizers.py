"""Optimizer registry/factory.

Add a new optimizer by writing a builder function and registering it via
`@register_optimizer("name")`. Builders receive an iterable of params, a base
`lr`, and an arbitrary `**kwargs` dict (parsed from `--optimizer-arg`); they
should consume what they recognize and pass it down.

Builders may return any `torch.optim.Optimizer`. Distributed/FSDP wiring is the
builder's responsibility — vanilla AdamW works on local shards out of the box,
but Shampoo-family optimizers need FSDP-aware variants (e.g. Meta's
`distributed-shampoo` package).
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch

OptimizerBuilder = Callable[..., torch.optim.Optimizer]

_REGISTRY: dict[str, OptimizerBuilder] = {}


def register_optimizer(name: str) -> Callable[[OptimizerBuilder], OptimizerBuilder]:
    def _decorator(fn: OptimizerBuilder) -> OptimizerBuilder:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Optimizer '{name}' is already registered")
        _REGISTRY[key] = fn
        return fn

    return _decorator


def available_optimizers() -> list[str]:
    return sorted(_REGISTRY.keys())


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {available_optimizers()}"
        )
    return _REGISTRY[key](params, lr=lr, **kwargs)


def _coerce(value: str) -> Any:
    if "," in value:
        return tuple(_coerce(v) for v in value.split(","))
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


@register_optimizer("adamw")
def _build_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    **_: Any,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )


@register_optimizer("adam")
def _build_adam(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    **_: Any,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )


@register_optimizer("sgd")
def _build_sgd(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    nesterov: bool = False,
    **_: Any,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )


@register_optimizer("lion")
def _build_lion(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.99),
    **_: Any,
) -> torch.optim.Optimizer:
    try:
        from lion_pytorch import Lion
    except ImportError as e:
        raise ImportError(
            "Lion requires the `lion-pytorch` package. "
            "Install with `pip install lion-pytorch`."
        ) from e
    return Lion(params, lr=lr, weight_decay=weight_decay, betas=betas)


@register_optimizer("shampoo")
def _build_shampoo(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-12,
    max_preconditioner_dim: int = 8192,
    precondition_frequency: int = 1,
    start_preconditioning_step: int = -1,
    use_decoupled_weight_decay: bool = True,
    grafting_type: str = "adam",
    **extra: Any,
) -> torch.optim.Optimizer:
    """Distributed Shampoo (Meta).

    Requires `pip install torch-optimizers` (Meta's `distributed-shampoo` package).
    For FSDP runs, callers should pass `distributed_config` via --optimizer-arg
    or rely on the package's auto-detection.
    """
    try:
        from distributed_shampoo import (  # type: ignore[import-not-found]
            AdamGraftingConfig,
            DistributedShampoo,
            SGDGraftingConfig,
        )
    except ImportError as e:
        raise ImportError(
            "Shampoo requires Meta's `distributed-shampoo` package. "
            "Install with `pip install torch-optimizers` "
            "(see https://github.com/facebookresearch/optimizers)."
        ) from e

    if grafting_type.lower() == "adam":
        grafting_config = AdamGraftingConfig(beta2=betas[1], epsilon=epsilon)
    elif grafting_type.lower() == "sgd":
        grafting_config = SGDGraftingConfig()
    elif grafting_type.lower() == "none":
        grafting_config = None
    else:
        raise ValueError(
            f"Unknown grafting_type '{grafting_type}'. "
            "Expected one of: adam, sgd, none."
        )

    return DistributedShampoo(
        params,
        lr=lr,
        betas=betas,
        epsilon=epsilon,
        weight_decay=weight_decay,
        max_preconditioner_dim=max_preconditioner_dim,
        precondition_frequency=precondition_frequency,
        start_preconditioning_step=start_preconditioning_step,
        use_decoupled_weight_decay=use_decoupled_weight_decay,
        grafting_config=grafting_config,
        **extra,
    )
