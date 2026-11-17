"""Affine coupling: forward maps data to latents; inverse maps latents to data."""

import math

import torch
from torch import nn


def checkerboard_mask(height: int, width: int, *, invert: bool = False) -> torch.Tensor:
    """Return a broadcastable (1, 1, H, W) mask; ones stay unchanged."""
    if height < 1 or width < 1:
        raise ValueError("spatial dimensions must be positive")
    rows = torch.arange(height)[:, None]
    columns = torch.arange(width)[None, :]
    mask = ((rows + columns) % 2).float()[None, None]
    return 1 - mask if invert else mask


def channel_mask(channels: int, *, invert: bool = False) -> torch.Tensor:
    """Keep the first half of channels; return shape (1, C, 1, 1)."""
    if channels < 2:
        raise ValueError("channel coupling requires at least two channels")
    mask = (torch.arange(channels) < channels // 2).float()[None, :, None, None]
    return 1 - mask if invert else mask


class CouplingNetwork(nn.Module):
    """Convolutional conditioner producing raw log-scale and translation.

    conditioner_depth counts all convolutions, including input and output.
    Depth 3 preserves the baseline 3x3 -> 1x1 -> 3x3 network and state-dict
    keys. Each extra layer is a hidden 3x3 convolution followed by ReLU,
    inserted before the output. Its theoretical receptive field at this
    scale is (2*conditioner_depth - 1) pixels per side: 5 for depth 3, 9 for 5.

    No stochastic or batch-dependent layers: inverse must reproduce the same
    parameters from the unchanged coordinates. Zero final initialization starts
    the coupling at identity while allowing its output layer to learn immediately.
    """

    def __init__(
        self, channels: int, hidden_channels: int = 64, *, conditioner_depth: int = 3,
    ) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("channel counts must be positive")
        if not isinstance(conditioner_depth, int) or conditioner_depth < 3:
            raise ValueError("conditioner_depth must be an integer of at least 3")
        self.conditioner_depth = conditioner_depth
        layers: list[nn.Module] = [
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
        ]
        for _ in range(conditioner_depth - 3):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
            ])
        layers.append(nn.Conv2d(hidden_channels, 2 * channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class AffineCoupling(nn.Module):
    """y = m*x + (1-m)*(x*exp(s(m*x)) + t(m*x)).

    The conditioner returns 2*C channels, split into raw scale and translation.
    s = scale_limit*tanh(raw_scale) bounds exponential factors. The Jacobian
    is triangular after ordering kept coordinates first, hence log|det J| is
    the sum of s on transformed coordinates. Both directions return (tensor,
    per-example log|det J|). Supports vectors with a suitable custom conditioner
    and mask as well as images, for reuse in a future latent-space flow.
    """

    def __init__(
        self, conditioner: nn.Module, mask: torch.Tensor, *, scale_limit: float = 2.0
    ) -> None:
        super().__init__()
        if mask.ndim < 2 or mask.shape[0] != 1:
            raise ValueError("mask must include a singleton batch dimension")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary")
        if not math.isfinite(scale_limit) or scale_limit <= 0:
            raise ValueError("scale_limit must be finite and positive")
        self.conditioner = conditioner
        self.register_buffer("mask", mask.detach().clone().float())
        self.scale_limit = scale_limit

    def _parameters_from(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != self.mask.ndim:
            raise ValueError("input and mask must have the same number of axes")
        if any(m not in (1, size) for m, size in zip(self.mask.shape[1:], inputs.shape[1:])):
            raise ValueError("mask is not broadcastable to the input")
        raw = self.conditioner(inputs * self.mask)
        expected = (inputs.shape[0], 2 * inputs.shape[1], *inputs.shape[2:])
        if raw.shape != expected:
            raise ValueError(f"conditioner output must have shape {expected}")
        raw_scale, translation = raw.chunk(2, dim=1)
        transformed = 1 - self.mask
        scale = self.scale_limit * torch.tanh(raw_scale) * transformed
        return scale, translation * transformed

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, translation = self._parameters_from(inputs)
        outputs = inputs * torch.exp(scale) + translation
        return outputs, scale.flatten(1).sum(1)

    def inverse(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, translation = self._parameters_from(outputs)
        inputs = (outputs - translation) * torch.exp(-scale)
        return inputs, -scale.flatten(1).sum(1)
