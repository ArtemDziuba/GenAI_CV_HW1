"""Dimension-preserving, multi-scale RealNVP for small images."""

import math

import torch
from torch import nn

from .coupling import AffineCoupling, CouplingNetwork, channel_mask, checkerboard_mask


def squeeze2d(inputs: torch.Tensor) -> torch.Tensor:
    """Permute (N,C,H,W) into (N,4*C,H/2,W/2); log|det J| is zero."""
    if inputs.ndim != 4:
        raise ValueError("squeeze expects (N, C, H, W)")
    batch, channels, height, width = inputs.shape
    if height % 2 or width % 2:
        raise ValueError("squeeze requires even height and width")
    return (
        inputs.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, channels * 4, height // 2, width // 2)
    )


def unsqueeze2d(inputs: torch.Tensor) -> torch.Tensor:
    """Invert squeeze2d exactly; no interpolation or padding."""
    if inputs.ndim != 4 or inputs.shape[1] % 4:
        raise ValueError("unsqueeze expects (N, C, H, W) with C divisible by four")
    batch, channels, height, width = inputs.shape
    return (
        inputs.reshape(batch, channels // 4, 2, 2, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, channels // 4, height * 2, width * 2)
    )


class FlowScale(nn.Module):
    """Checkerboard couplings, squeeze, then channel couplings."""

    def __init__(
        self, shape: tuple[int, int, int], hidden_channels: int,
        couplings_per_scale: int, scale_limit: float, conditioner_depth: int = 3,
    ) -> None:
        super().__init__()
        channels, height, width = shape
        self.checkerboard = nn.ModuleList([
            AffineCoupling(
                CouplingNetwork(channels, hidden_channels, conditioner_depth=conditioner_depth),
                checkerboard_mask(height, width, invert=bool(index % 2)),
                scale_limit=scale_limit,
            )
            for index in range(couplings_per_scale)
        ])
        self.channel = nn.ModuleList([
            AffineCoupling(
                CouplingNetwork(4 * channels, hidden_channels, conditioner_depth=conditioner_depth),
                channel_mask(4 * channels, invert=bool(index % 2)),
                scale_limit=scale_limit,
            )
            for index in range(couplings_per_scale)
        ])

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = inputs.new_zeros(inputs.shape[0])
        for coupling in self.checkerboard:
            inputs, update = coupling(inputs)
            log_det = log_det + update
        inputs = squeeze2d(inputs)
        for coupling in self.channel:
            inputs, update = coupling(inputs)
            log_det = log_det + update
        return inputs, log_det

    def inverse(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = outputs.new_zeros(outputs.shape[0])
        for coupling in reversed(self.channel):
            outputs, update = coupling.inverse(outputs)
            log_det = log_det + update
        outputs = unsqueeze2d(outputs)
        for coupling in reversed(self.checkerboard):
            outputs, update = coupling.inverse(outputs)
            log_det = log_det + update
        return outputs, log_det


class RealNVP(nn.Module):
    """Factor out half the channels after each nonfinal scale.

    Latent order is [first factored block, ..., final active block], flattened
    per image. Inverse starts with the final block, then reattaches factored
    blocks in reverse order before undoing each preceding scale.
    Default 28x28 topology: 1x28x28 -> 4x14x14 -> factor 2x14x14 (392),
    continue 2x14x14 -> 8x7x7 (392). Total latent size remains 784.
    conditioner_depth changes the CNN inside every coupling, independently of
    the number of coupling transformations. Default 3 loads legacy checkpoints.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 28, 28),
        *,
        hidden_channels: int = 64,
        num_scales: int = 2,
        couplings_per_scale: int = 3,
        scale_limit: float = 2.0,
        conditioner_depth: int = 3,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(size < 1 for size in input_shape):
            raise ValueError("input_shape must contain three positive dimensions")
        if num_scales < 1 or couplings_per_scale < 1:
            raise ValueError("num_scales and couplings_per_scale must be positive")
        if not isinstance(conditioner_depth, int) or conditioner_depth < 3:
            raise ValueError("conditioner_depth must be an integer of at least 3")
        channels, height, width = input_shape
        if height % 2**num_scales or width % 2**num_scales:
            raise ValueError("image height and width must be divisible by 2**num_scales")
        self.input_shape = tuple(input_shape)
        self.dimensions = math.prod(input_shape)
        self.conditioner_depth = conditioner_depth
        self.scales = nn.ModuleList()
        self.latent_shapes: list[tuple[int, int, int]] = []
        for index in range(num_scales):
            self.scales.append(FlowScale(
                (channels, height, width), hidden_channels, couplings_per_scale, scale_limit,
                conditioner_depth=conditioner_depth,
            ))
            channels, height, width = 4 * channels, height // 2, width // 2
            if index < num_scales - 1:
                channels //= 2
            self.latent_shapes.append((channels, height, width))
        self.latent_sizes = [math.prod(shape) for shape in self.latent_shapes]
        assert sum(self.latent_sizes) == self.dimensions

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f"images must have shape (N, {self.input_shape})")
        log_det = images.new_zeros(images.shape[0])
        blocks = []
        active = images
        for index, scale in enumerate(self.scales):
            active, update = scale(active)
            log_det = log_det + update
            if index < len(self.scales) - 1:
                active, factored = active.chunk(2, dim=1)
                blocks.append(factored.flatten(1))
        blocks.append(active.flatten(1))
        return torch.cat(blocks, dim=1), log_det

    def inverse(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 2 or latents.shape[1] != self.dimensions:
            raise ValueError(f"latents must have shape (N, {self.dimensions})")
        flat_blocks = latents.split(self.latent_sizes, dim=1)
        blocks = [block.reshape(-1, *shape) for block, shape in zip(
            flat_blocks, self.latent_shapes, strict=True
        )]
        active = blocks[-1]
        log_det = latents.new_zeros(latents.shape[0])
        for index in reversed(range(len(self.scales))):
            if index < len(self.scales) - 1:
                active = torch.cat((active, blocks[index]), dim=1)
            active, update = self.scales[index].inverse(active)
            log_det = log_det + update
        return active, log_det
