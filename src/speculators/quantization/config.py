"""
Base configuration classes for weight/activation quantization of speculator models using QuEST.
"""

from abc import abstractmethod
from typing import ClassVar

import torch.nn as nn
from pydantic import Field

from speculators.utils import PydanticClassRegistryMixin
from speculators.utils.pydantic_utils import ReloadableBaseModel

__all__ = ["QuantizationConfig", "QuantizerConfig"]


class QuantizerConfig(PydanticClassRegistryMixin):
    """Polymorphic config for a single quantizer (weight or activation)."""

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type["QuantizerConfig"]:
        if cls.__name__ == "QuantizerConfig":
            return cls
        return QuantizerConfig

    auto_package: ClassVar[str] = "speculators.quantization"
    registry_auto_discovery: ClassVar[bool] = True
    schema_discriminator: ClassVar[str] = "quantizer_type"

    quantizer_type: str = Field(
        description=(
            "The quantizer implementation key. Must match a registered "
            "QuantizerConfig subclass."
        ),
    )

    @abstractmethod
    def build(self) -> nn.Module:
        """Instantiate the underlying quantizer ``nn.Module``."""
        ...


class QuantizationConfig(ReloadableBaseModel):
    """Container for weight + activation quantizer configs and target patterns.

    Held by speculator configs under a ``quantization`` field. ``None`` means
    no quantization (default behavior preserved).

    Targets/excludes are Python regex patterns matched against dotted module
    paths from ``model.named_modules()`` (relative to the speculator model
    root). Defaults quantize all attention and MLP projections inside decoder
    layers.
    """

    weight: QuantizerConfig = Field(
        default_factory=lambda: _default_no_quantizer(),
        description="Weight quantizer configuration.",
    )
    activation: QuantizerConfig = Field(
        default_factory=lambda: _default_no_quantizer(),
        description="Activation quantizer configuration.",
    )
    targets: list[str] = Field(
        default_factory=lambda: [r"^layers\.\d+\.(self_attn|mlp)\..*_proj$"],
        description=(
            "Regex patterns matching dotted module paths to quantize. "
            "A module is quantized if it matches at least one target pattern "
            "and no exclude pattern."
        ),
    )
    # excludes: list[str] = Field(
    #     default_factory=lambda: [r"^layers\.0\.self_attn\.(q|k|v)_proj$"],
    #     description=(
    #         "Regex patterns matching dotted module paths to skip. Defaults "
    #         "exclude the Eagle3 first-layer q/k/v projections (2*hidden_size "
    #         "input) which mix embed and hidden distributions."
    #     ),
    # )


def _default_no_quantizer() -> QuantizerConfig:
    # Lazy import: quest module imports QuantizerConfig from this module, so
    # importing it at module top-level would create a circular import during
    # registry auto-discovery.
    from speculators.quantization.quest import NoQuantizerConfig

    return NoQuantizerConfig()
