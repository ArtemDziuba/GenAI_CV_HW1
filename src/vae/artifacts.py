"""Comparable reconstructions, prior draws, and optional latent traversals."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from shared import plot_image_grid, plot_metrics

from .models import Autoencoder, VAE
from .objectives import prepare_observations


@torch.no_grad()
def save_reconstructions(
    model: Autoencoder | VAE, images: torch.Tensor, path: Path, *,
    likelihood: str = "bernoulli", seed: int = 42, count: int = 24, nrow: int = 8,
) -> None:
    """Save comparison cells and a reconstruction-only companion grid.

    Each comparison cell stacks original, optional binary observation, and
    base-mean reconstruction vertically. Default: 24 cells in three rows of
    eight. The companion ``<stem>_only.png`` shows just the 24 reconstructions.
    """
    if count < 1 or nrow < 1 or images.shape[0] < 1:
        raise ValueError("count, nrow, and image batch size must be positive")
    model.eval()
    images = images[:count].to(next(model.parameters()).device)
    rows = [images]
    if isinstance(model, Autoencoder):
        reconstruction = model(images)
    else:
        generator = torch.Generator(device=images.device).manual_seed(seed)
        observed = prepare_observations(images, likelihood, generator=generator)
        if likelihood == "bernoulli":
            rows.append(observed)
        mean, _ = model.encode(observed)
        latent, _ = model.posterior_flow(mean)
        reconstruction = model.decode(latent)
    rows.append(reconstruction)
    separator = images.new_full((*images.shape[:2], 2, images.shape[-1]), 0.5)
    panels = [rows[0]]
    for row in rows[1:]:
        panels.extend([separator, row])
    batch = torch.cat(panels, dim=2)
    figure = plot_image_grid(
        batch, nrow=nrow, max_images=len(batch), save_path=path,
        title="Each cell: original / binary input / reconstruction" if len(rows) == 3 else "Each cell: original / reconstruction",
    )
    plt.close(figure)
    figure = plot_image_grid(
        reconstruction, nrow=nrow, max_images=len(images),
        save_path=path.with_name(f"{path.stem}_only{path.suffix}"),
        title=f"{len(images)} reconstructions",
    )
    plt.close(figure)


@torch.no_grad()
def save_samples(
    model: VAE, path: Path, *, seed: int = 42, count: int = 64,
    temperature: float = 1.0, likelihood: str = "bernoulli", observation_std: float = 0.1,
    sample_observations: bool = False,
) -> None:
    model.eval()
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    means = model.sample(count, generator=generator, temperature=temperature)
    samples = means
    if sample_observations:
        if likelihood == "bernoulli":
            samples = torch.bernoulli(means, generator=generator)
        else:
            noise = torch.randn(means.shape, device=means.device, dtype=means.dtype, generator=generator)
            samples = means + observation_std * noise
    figure = plot_image_grid(
        samples, max_images=count, save_path=path,
        title=f"Prior T={temperature:g}: {'observation draws' if sample_observations else 'decoder means'}",
    )
    plt.close(figure)


@torch.no_grad()
def save_traversal(
    model: VAE, path: Path, *, dimensions: int = 8, steps: int = 9, extent: float = 3.0,
) -> None:
    """Each row varies one prior coordinate, holding all others at zero.

    This visualizes sensitivity; it does not measure/guarantee disentanglement.
    Posterior flow is intentionally absent from prior-space traversal.
    """
    if dimensions < 1 or steps < 2 or not 0 < extent < float("inf"):
        raise ValueError("dimensions positive, steps >= 2, extent finite and positive required")
    model.eval()
    parameter = next(model.parameters())
    dimensions = min(dimensions, model.latent_dim)
    latents = parameter.new_zeros(dimensions * steps, model.latent_dim)
    values = torch.linspace(-extent, extent, steps, device=parameter.device, dtype=parameter.dtype)
    for index in range(dimensions):
        latents[index * steps:(index + 1) * steps, index] = values
    figure = plot_image_grid(
        model.decode(latents), nrow=steps, max_images=len(latents), save_path=path,
        title=f"Prior coordinate traversals: {-extent:g} to {extent:g}",
    )
    plt.close(figure)


def save_history(history: list[dict], output: Path, *, variational: bool) -> None:
    metrics = ["objective", "grayscale_mse"]
    if "grayscale_mae" in history[0]["train"]:
        metrics.append("grayscale_mae")
    if variational:
        metrics.extend(["reconstruction_nll", "kl", "negative_elbo"])
    for metric in metrics:
        figure = plot_metrics(
            {split: [row[split][metric] for row in history] for split in ("train", "validation")},
            ylabel=(f"{metric.removeprefix('grayscale_').upper()} / pixel" if metric.startswith("grayscale_")
                    else "Sum / image" if not variational else "Nats / image"),
            title=metric, save_path=output / f"{metric}.png",
        )
        plt.close(figure)
