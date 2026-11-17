"""Uniform dequantization and explicit continuous/discrete likelihood conventions."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .realnvp import RealNVP


def uniform_dequantize(
    images: torch.Tensor, *, bits: int = 8,
    generator: torch.Generator | None = None, noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover integer q from q/(2**bits-1), then return (q+U)/2**bits.

    Inputs must be normalized discrete images, such as torchvision.ToTensor()
    MNIST. Rounding recovers the original integer; arbitrary continuous inputs
    would be quantized. Noise is U[0,1), independent per pixel. This operation
    does not itself apply the -D*log(2**bits) discrete bin correction.
    """
    if not isinstance(bits, int) or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer between 1 and 16")
    if images.ndim < 2 or not images.is_floating_point():
        raise ValueError("images must be floating-point tensors with a batch axis")
    if not torch.isfinite(images).all() or torch.any((images < 0) | (images > 1)):
        raise ValueError("normalized image values must be finite and in [0, 1]")
    if noise is None:
        noise = torch.rand(images.shape, dtype=images.dtype, device=images.device, generator=generator)
    elif noise.shape != images.shape or noise.device != images.device:
        raise ValueError("noise must have the same shape and device as images")
    if not torch.isfinite(noise).all() or torch.any((noise < 0) | (noise >= 1)):
        raise ValueError("noise must be finite and in [0, 1)")
    levels = 2**bits
    return (torch.round(images * (levels - 1)) + noise) / levels


class ImagePreprocessing(nn.Module):
    """Optional contracted logit on dequantized u; no clipping in either direction.

    v = alpha + (1-2*alpha)*u, y = log(v)-log(1-v).
    log|dy/du| = log(1-2*alpha)-log(v)-log(1-v), summed per image.
    With logit=False this is the identity. Inverse Gaussian samples may fall
    outside the data cube; clamp only for display, not for likelihood/inversion.
    """

    def __init__(self, *, logit: bool = True, alpha: float = 1e-5) -> None:
        super().__init__()
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5")
        self.logit = logit
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.logit:
            return inputs, inputs.new_zeros(inputs.shape[0])
        contracted = self.alpha + (1 - 2 * self.alpha) * inputs
        if torch.any((contracted <= 0) | (contracted >= 1)):
            raise ValueError("logit inputs must lie inside the contracted domain")
        log_v, log_one_minus_v = contracted.log(), torch.log1p(-contracted)
        outputs = log_v - log_one_minus_v
        log_det = math.log1p(-2 * self.alpha) - log_v - log_one_minus_v
        return outputs, log_det.flatten(1).sum(1)

    def inverse(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.logit:
            return outputs, outputs.new_zeros(outputs.shape[0])
        inputs = (torch.sigmoid(outputs) - self.alpha) / (1 - 2 * self.alpha)
        inverse_log_det = -math.log1p(-2 * self.alpha) - F.softplus(-outputs) - F.softplus(outputs)
        return inputs, inverse_log_det.flatten(1).sum(1)


def standard_normal_log_prob(latents: torch.Tensor) -> torch.Tensor:
    """Log density under an independent unit Gaussian, summed over dimensions."""
    return -0.5 * (latents.square() + math.log(2 * math.pi)).flatten(1).sum(1)


def negative_log_likelihood(log_prob: torch.Tensor) -> torch.Tensor:
    """Mean NLL in nats per image from per-image log probabilities."""
    return -log_prob.mean()


def bits_per_dimension(log_prob: torch.Tensor, dimensions: int) -> torch.Tensor:
    """Per-image BPD; log_prob must already include the discrete bin correction."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    return -log_prob / (dimensions * math.log(2))


class RealNVPDensity(nn.Module):
    """Combine flow, preprocessing, and prior, including the 8-bit bin volume.

    log_prob returns one stochastic estimate per image:
    log p(z) + log|dz/dy| + log|dy/du| - D*log(2**bits).
    Its expectation over uniform dequantization lower-bounds discrete log mass;
    negative log_prob and BPD are upper-bound estimates of discrete NLL/BPD.
    This is not exact discrete MNIST likelihood or an importance-sampled estimate.
    """

    def __init__(
        self, flow: RealNVP, *, logit: bool = True, alpha: float = 1e-5, bits: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(bits, int) or not 1 <= bits <= 16:
            raise ValueError("bits must be an integer between 1 and 16")
        self.flow = flow
        self.preprocessing = ImagePreprocessing(logit=logit, alpha=alpha)
        self.bits = bits

    def log_prob(
        self, images: torch.Tensor, *, generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unit_images = uniform_dequantize(images, bits=self.bits, generator=generator, noise=noise)
        transformed, preprocess_log_det = self.preprocessing(unit_images)
        latents, flow_log_det = self.flow(transformed)
        return (
            standard_normal_log_prob(latents) + flow_log_det + preprocess_log_det
            - self.flow.dimensions * self.bits * math.log(2)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-image dequantized log probabilities, like log_prob."""
        return self.log_prob(images)

    @torch.no_grad()
    def sample(self, count: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw continuous images; retain out-of-range values until display."""
        if count < 1:
            raise ValueError("sample count must be positive")
        parameter = next(self.parameters())
        latents = torch.randn(
            count, self.flow.dimensions, device=parameter.device, dtype=parameter.dtype,
            generator=generator,
        )
        transformed, _ = self.flow.inverse(latents)
        images, _ = self.preprocessing.inverse(transformed)
        return images
