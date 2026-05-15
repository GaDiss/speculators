from typing import Literal

import torch.nn as nn
from pydantic import Field

from speculators.quantization.config import QuantizerConfig
from speculators.quantization.base_linear import QUANTIZER_CLASSES

__all__ = [
    "HadamardTrustQuantizerConfig",
    "NoQuantizerConfig",
    "STEQuantizerConfig",
    "TrustQuantizerConfig",
]


@QuantizerConfig.register("no_quantizer")
class NoQuantizerConfig(QuantizerConfig):
    quantizer_type: Literal["no_quantizer"] = "no_quantizer"

    def build(self) -> nn.Module:
        return QUANTIZER_CLASSES["NoQuantizer"]()


@QuantizerConfig.register("ste")
class STEQuantizerConfig(QuantizerConfig):
    quantizer_type: Literal["ste"] = "ste"
    bits: int = Field(default=4, ge=1, le=16)
    centered: bool = True

    def build(self) -> nn.Module:
        return QUANTIZER_CLASSES["STEQuantizer"](
            bits=self.bits, centered=self.centered,
        )


@QuantizerConfig.register("trust")
class TrustQuantizerConfig(QuantizerConfig):
    quantizer_type: Literal["trust"] = "trust"
    bits: int = Field(default=4, ge=1, le=16)
    centered: bool = True
    trust: float | None = None

    def build(self) -> nn.Module:
        return QUANTIZER_CLASSES["TrustQuantizer"](
            bits=self.bits, centered=self.centered, trust=self.trust,
        )


@QuantizerConfig.register("hadamard_trust")
class HadamardTrustQuantizerConfig(QuantizerConfig):
    quantizer_type: Literal["hadamard_trust"] = "hadamard_trust"
    bits: int = Field(default=4, ge=1, le=16)
    trust: float | None = None

    def build(self) -> nn.Module:
        return QUANTIZER_CLASSES["HadamardTrustQuantizer"](
            bits=self.bits, trust=self.trust,
        )
    
