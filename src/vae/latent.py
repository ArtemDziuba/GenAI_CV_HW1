"""Diagonal Gaussian densities and a vector RealNVP posterior flow."""

import math

import torch
from torch import nn

from flows.coupling import AffineCoupling


def reparameterize(
    mean: torch.Tensor, log_variance: torch.Tensor, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pathwise draw z = mu + exp(logvar/2) * epsilon; preserves both gradients."""
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean + torch.exp(0.5 * log_variance) * noise


def diagonal_normal_log_prob(
    value: torch.Tensor, mean: torch.Tensor, log_variance: torch.Tensor,
) -> torch.Tensor:
    """Log density summed over latent coordinates, in nats per example."""
    return -0.5 * (
        math.log(2 * math.pi) + log_variance + (value - mean).square() * torch.exp(-log_variance)
    ).sum(-1)


def standard_normal_kl(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Exact KL[N(mu, diag(exp(logvar))) || N(0,I)], summed over coordinates."""
    return 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(-1)


class LatentFlow(nn.Module):
    """Shared posterior transformation z0 -> zK, with alternating vector masks.

    This transforms the approximate posterior, not the generative prior. The
    conditioners depend on masked z; their weights are shared across images.
    The base posterior's mean/variance depend on x. No image-scale squeeze or
    factor-out is needed in this fixed-dimensional latent space.
    """

    def __init__(
        self, latent_dim: int, steps: int = 4, hidden_dim: int = 128,
        hidden_layers: int = 2, scale_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or steps < 0 or hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("latent/hidden dimensions and hidden_layers must be positive; steps nonnegative")
        if steps and latent_dim < 2:
            raise ValueError("a coupling posterior requires latent_dim >= 2")
        if not math.isfinite(scale_limit) or scale_limit <= 0:
            raise ValueError("scale_limit must be finite and positive")
        self.layers = nn.ModuleList()
        for index in range(steps):
            layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
            for _ in range(hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            output = nn.Linear(hidden_dim, 2 * latent_dim)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            layers.append(output)
            mask = ((torch.arange(latent_dim) + index) % 2).float()[None]
            self.layers.append(AffineCoupling(nn.Sequential(*layers), mask, scale_limit=scale_limit))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = inputs.new_zeros(inputs.shape[0])
        for layer in self.layers:
            inputs, contribution = layer(inputs)
            log_det = log_det + contribution
        return inputs, log_det

    def inverse(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = outputs.new_zeros(outputs.shape[0])
        for layer in reversed(self.layers):
            outputs, contribution = layer.inverse(outputs)
            log_det = log_det + contribution
        return outputs, log_det
