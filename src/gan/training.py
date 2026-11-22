"""Separate GAN optimization phases and repeatable validation diagnostics."""

from collections.abc import Callable
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import Discriminator, Generator


@contextmanager
def frozen(module: nn.Module):
    """Freeze parameters AND BatchNorm buffers, retaining input gradients."""
    flags = [parameter.requires_grad for parameter in module.parameters()]
    training = module.training
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)
        module.train(training)


def discriminator_terms(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean BCE over images and patches; real and fake terms weighted 1/2."""
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return {"d_loss": 0.5 * (real_loss + fake_loss), "d_real_loss": real_loss, "d_fake_loss": fake_loss,
            "d_real_probability": real_logits.sigmoid().mean(),
            "d_fake_probability": fake_logits.sigmoid().mean(),
            "d_accuracy": 0.5 * ((real_logits >= 0).float().mean() + (fake_logits < 0).float().mean())}


def feature_matching(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """MSE between batch-mean penultimate features, averaged over C,H,W.

    The real target is detached; gradients through fake features reach G. For
    conditional models this matches marginal batch features, not classwise means.
    """
    return (real.detach().mean(0) - fake.mean(0)).square().mean()


def _metrics(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    if not all(torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("nonfinite GAN objective or diagnostic")
    return {key: value.detach().item() for key, value in terms.items()}


def discriminator_step(generator: Generator, discriminator: Discriminator, real: torch.Tensor,
                       labels: torch.Tensor | None, noise: torch.Tensor,
                       optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Only D changes; G runs in eval/no_grad so its BatchNorm is untouched."""
    generator.zero_grad(set_to_none=True)
    discriminator.train()
    optimizer.zero_grad(set_to_none=True)
    with frozen(generator), torch.no_grad():
        fake = generator(noise, labels)
    terms = discriminator_terms(discriminator(real, labels), discriminator(fake, labels))
    metrics = _metrics(terms)
    terms["d_loss"].backward()
    optimizer.step()
    return metrics


def generator_step(generator: Generator, discriminator: Discriminator, real: torch.Tensor,
                   labels: torch.Tensor | None, noise: torch.Tensor,
                   optimizer: torch.optim.Optimizer, *, feature_matching_weight: float = 0) -> dict[str, float]:
    """Only G changes; D retains input gradients but no parameter/buffer updates.

    The generator uses non-saturating BCE, optionally PLUS lambda * feature MSE.
    Feature matching does not silently replace the adversarial objective.
    """
    if not math.isfinite(feature_matching_weight) or feature_matching_weight < 0:
        raise ValueError("feature matching weight must be finite and nonnegative")
    generator.train()
    optimizer.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    with frozen(discriminator):
        fake = generator(noise, labels)
        fake_logits, fake_features = discriminator(fake, labels, return_features=True)
        adversarial = F.softplus(-fake_logits).mean()
        matching = adversarial.new_zeros(())
        if feature_matching_weight:
            with torch.no_grad():
                _, real_features = discriminator(real, labels, return_features=True)
            matching = feature_matching(real_features, fake_features)
        loss = adversarial + feature_matching_weight * matching
        terms = {"g_loss": loss, "g_adversarial_loss": adversarial, "g_feature_matching_loss": matching,
                 "g_fake_probability": fake_logits.sigmoid().mean()}
        metrics = _metrics(terms)
        loss.backward()
    optimizer.step()
    return metrics


def run_epoch(generator: Generator, discriminator: Discriminator, loader: DataLoader, device: torch.device, *,
              generator_optimizer: torch.optim.Optimizer | None = None,
              discriminator_optimizer: torch.optim.Optimizer | None = None,
              discriminator_steps: int = 1, feature_matching_weight: float = 0, seed: int = 42,
              progress: bool = False, on_step: Callable[[int, dict], None] | None = None) -> dict[str, float]:
    """Image-weighted epoch means, including the final partial batch.

    Raw [0,1] pixels are mapped to [-1,1] here, not in the shared dataset loader.
    Validation freezes both models, with a fixed local noise seed. GAN validation
    losses/accuracy describe the current adversarial game, not image quality.
    """
    if (generator_optimizer is None) != (discriminator_optimizer is None):
        raise ValueError("provide both optimizers for training, or neither for evaluation")
    if discriminator_steps < 1 or not math.isfinite(feature_matching_weight) or feature_matching_weight < 0:
        raise ValueError("D steps must be positive; feature matching weight finite and nonnegative")
    if generator.conditional != discriminator.conditional or generator.num_classes != discriminator.num_classes:
        raise ValueError("G and D conditioning must agree")
    training = generator_optimizer is not None
    generator.train(training)
    discriminator.train(training)
    rng = torch.Generator(device=device).manual_seed(seed)
    totals, count, batches = {}, 0, 0
    for batch_index, (images, targets) in enumerate(tqdm(loader, disable=not progress, desc="train" if training else "validation", leave=False), 1):
        images = images.to(device, non_blocking=True)
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("GAN loader must return finite original grayscale pixels in [0,1]")
        real = images.mul(2).sub(1)
        labels = targets.to(device) if generator.conditional else None
        def noise():
            return torch.randn(len(real), generator.latent_dim, device=device, generator=rng)
        if training:
            d_rows = [discriminator_step(generator, discriminator, real, labels, noise(), discriminator_optimizer)
                      for _ in range(discriminator_steps)]
            metrics = {key: sum(row[key] for row in d_rows) / len(d_rows) for key in d_rows[0]}
            metrics.update(generator_step(generator, discriminator, real, labels, noise(), generator_optimizer,
                                          feature_matching_weight=feature_matching_weight))
        else:
            with torch.no_grad():
                fake = generator(noise(), labels)
                real_logits, real_features = discriminator(real, labels, return_features=True)
                fake_logits, fake_features = discriminator(fake, labels, return_features=True)
                terms = discriminator_terms(real_logits, fake_logits)
                adversarial = F.softplus(-fake_logits).mean()
                matching = feature_matching(real_features, fake_features) if feature_matching_weight else adversarial.new_zeros(())
                terms.update(g_loss=adversarial + feature_matching_weight * matching,
                             g_adversarial_loss=adversarial, g_feature_matching_loss=matching,
                             g_fake_probability=fake_logits.sigmoid().mean())
                metrics = _metrics(terms)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value * len(real)
        count += len(real)
        batches += 1
        if on_step is not None:
            on_step(batch_index, {**metrics, "batch_images": len(real)})
    if not count:
        raise ValueError("loader must be nonempty")
    return {**{key: value / count for key, value in totals.items()}, "images": count, "batches": batches,
            "g_updates": batches if training else 0, "d_updates": batches * discriminator_steps if training else 0}
