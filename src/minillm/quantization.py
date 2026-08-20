"""Reference quantization-aware-training modules for deployment experiments."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QATConfig:
    weight_bits: int = 4
    activation_bits: int = 8
    group_size: int = 32

    def validate(self) -> QATConfig:
        if not 2 <= self.weight_bits <= 8 or not 4 <= self.activation_bits <= 16:
            raise ValueError("unsupported quantization precision")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        return self


def fake_quantize_groupwise(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    straight_through: bool = True,
) -> torch.Tensor:
    """Symmetric per-group fake quantization along the final dimension."""

    if not tensor.is_floating_point():
        raise TypeError("fake quantization expects floating-point input")
    if bits < 2 or group_size < 1:
        raise ValueError("invalid fake quantization settings")
    original_shape = tensor.shape
    width = original_shape[-1]
    padding = (-width) % group_size
    padded = F.pad(tensor, (0, padding)) if padding else tensor
    groups = padded.reshape(*padded.shape[:-1], -1, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = groups.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(groups / scale).clamp(-qmax, qmax) * scale
    restored = quantized.reshape(*padded.shape)[..., :width].reshape(original_shape)
    return tensor + (restored - tensor).detach() if straight_through else restored


def fake_quantize_activations(tensor: torch.Tensor, *, bits: int = 8) -> torch.Tensor:
    """Dynamic symmetric per-token activation fake quantization."""

    qmax = 2 ** (bits - 1) - 1
    scale = tensor.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax) * scale
    return tensor + (quantized - tensor).detach()


DEFAULT_QAT_CONFIG = QATConfig()


class QATLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        config: QATConfig,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config.validate()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, layer: nn.Linear, config: QATConfig) -> QATLinear:
        converted = cls(
            layer.in_features, layer.out_features, layer.bias is not None, config
        )
        # Reuse Parameter objects to preserve tied weights and optimizer identity.
        converted.weight = layer.weight
        if layer.bias is not None:
            converted.bias = layer.bias
        return converted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_input = fake_quantize_activations(x, bits=self.config.activation_bits)
        quantized_weight = fake_quantize_groupwise(
            self.weight,
            bits=self.config.weight_bits,
            group_size=self.config.group_size,
        )
        return F.linear(quantized_input, quantized_weight, self.bias)


def prepare_qat(
    module: nn.Module,
    config: QATConfig = DEFAULT_QAT_CONFIG,
    *,
    skip: tuple[str, ...] = ("*.router",),
    _prefix: str = "",
) -> nn.Module:
    """Recursively replace Linear layers while honoring full-name glob exclusions."""

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if isinstance(child, nn.Linear) and not any(
            fnmatch.fnmatch(full_name, pattern) for pattern in skip
        ):
            setattr(module, name, QATLinear.from_linear(child, config))
        else:
            prepare_qat(child, config, skip=skip, _prefix=full_name)
    return module
