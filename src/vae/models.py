"""Convolutional AE and VAE models for 1x28x28 images."""

from dataclasses import dataclass

import torch
from torch import nn

from .latent import LatentFlow, diagonal_normal_log_prob, reparameterize


class Encoder(nn.Module):
    """28 -> 14 -> 7 spatial resolution, then a configurable feature vector."""

    def __init__(self, hidden_channels: int, hidden_dim: int, conv_depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 1
        for width in (hidden_channels, 2 * hidden_channels):
            layers.extend([nn.Conv2d(channels, width, 4, stride=2, padding=1), nn.SiLU()])
            for _ in range(conv_depth - 1):
                layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.SiLU()])
            channels = width
        layers.extend([nn.Flatten(), nn.Linear(channels * 7 * 7, hidden_dim), nn.SiLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (N, 1, 28, 28)")
        return self.net(images)


class Decoder(nn.Module):
    """Latents -> 7 -> 14 -> 28; outputs unconstrained observation logits."""

    def __init__(self, latent_dim: int, hidden_channels: int, hidden_dim: int, conv_depth: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_channels * 7 * 7), nn.SiLU(),
            nn.Unflatten(1, (2 * hidden_channels, 7, 7)),
        )
        layers: list[nn.Module] = []
        channels = 2 * hidden_channels
        for stage, width in enumerate((hidden_channels, 1)):
            for _ in range(conv_depth - 1):
                layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU()])
            layers.append(nn.ConvTranspose2d(channels, width, 4, stride=2, padding=1))
            if stage == 0:
                layers.append(nn.SiLU())
            channels = width
        self.net = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(self.project(latents))


def _validate_architecture(latent_dim: int, hidden_channels: int, hidden_dim: int, conv_depth: int) -> None:
    if any(not isinstance(value, int) or value < 1
           for value in (latent_dim, hidden_channels, hidden_dim, conv_depth)):
        raise ValueError("architecture dimensions and conv_depth must be positive integers")


class Autoencoder(nn.Module):
    """Deterministic bottleneck trained by summed squared or absolute reconstruction error.

    No prior is fitted: N(0,I) samples are not justified for this plain AE.
    """

    def __init__(
        self, latent_dim: int = 16, hidden_channels: int = 32, hidden_dim: int = 256,
        conv_depth: int = 1,
    ) -> None:
        super().__init__()
        _validate_architecture(latent_dim, hidden_channels, hidden_dim, conv_depth)
        self.latent_dim = latent_dim
        self.encoder = Encoder(hidden_channels, hidden_dim, conv_depth)
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, hidden_dim, conv_depth)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.encoder(images))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents).sigmoid()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


@dataclass
class VAEOutput:
    logits: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor
    base_latent: torch.Tensor
    latent: torch.Tensor
    log_det: torch.Tensor
    log_q: torch.Tensor
    log_prior: torch.Tensor


class VAE(nn.Module):
    """Diagonal Gaussian VAE, optionally with an affine-coupling posterior.

    For flow_steps > 0: qK(zK|x) = q0(z0|x) / |det dzK/dz0|.
    The generative model always uses p(z)=N(0,I); generate by decoding that
    prior directly. Decoder sigmoid gives Bernoulli probabilities or the mean
    of a fixed-variance Gaussian, depending on the chosen observation model.
    """

    def __init__(
        self, latent_dim: int = 16, hidden_channels: int = 32, hidden_dim: int = 256,
        conv_depth: int = 1, flow_steps: int = 0, flow_hidden_dim: int = 128,
        flow_hidden_layers: int = 2, scale_limit: float = 1.0,
    ) -> None:
        super().__init__()
        _validate_architecture(latent_dim, hidden_channels, hidden_dim, conv_depth)
        self.latent_dim = latent_dim
        self.encoder = Encoder(hidden_channels, hidden_dim, conv_depth)
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, hidden_dim, conv_depth)
        self.posterior_flow = LatentFlow(
            latent_dim, flow_steps, flow_hidden_dim, flow_hidden_layers, scale_limit,
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        # Explicit variance bound avoids exp overflow; documented experiment convention.
        return self.mean_head(features), self.log_variance_head(features).clamp(-12, 12)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Observation means/probabilities, without additional observation noise."""
        return self.decoder(latents).sigmoid()

    def forward(
        self, images: torch.Tensor, *, generator: torch.Generator | None = None,
    ) -> VAEOutput:
        mean, log_variance = self.encode(images)
        base = reparameterize(mean, log_variance, generator=generator)
        latent, log_det = self.posterior_flow(base)
        log_q = diagonal_normal_log_prob(base, mean, log_variance) - log_det
        log_prior = diagonal_normal_log_prob(latent, torch.zeros_like(latent), torch.zeros_like(latent))
        return VAEOutput(self.decoder(latent), mean, log_variance, base, latent, log_det, log_q, log_prior)

    @torch.no_grad()
    def sample(
        self, count: int, *, generator: torch.Generator | None = None, temperature: float = 1.0,
    ) -> torch.Tensor:
        """Decode prior draws; temperature != 1 is a visualization experiment."""
        if count < 1 or not 0 < temperature < float("inf"):
            raise ValueError("count and temperature must be positive and finite")
        parameter = next(self.parameters())
        latents = torch.randn(
            count, self.latent_dim, device=parameter.device, dtype=parameter.dtype, generator=generator,
        ) * temperature
        return self.decode(latents)


def build_model(kind: str, **configuration: int | float) -> Autoencoder | VAE:
    """Factory used by CLI and checkpoints; fvae explicitly requires a nonempty flow."""
    if kind == "ae":
        return Autoencoder(**configuration)
    if kind not in ("vae", "fvae"):
        raise ValueError("model must be ae, vae, or fvae")
    if kind == "vae" and configuration.get("flow_steps", 0) != 0:
        raise ValueError("use model=fvae for a transformed posterior")
    if kind == "fvae" and configuration.get("flow_steps", 0) < 1:
        raise ValueError("fvae requires at least one flow step")
    return VAE(**configuration)
