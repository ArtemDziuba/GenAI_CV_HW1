"""Explicit observation distributions and AE / (beta-)ELBO conventions."""

import math

import torch
from torch.nn import functional as F

from .latent import standard_normal_kl
from .models import VAEOutput


def validate_likelihood(likelihood: str, observation_std: float) -> None:
    if likelihood not in ("bernoulli", "gaussian"):
        raise ValueError("likelihood must be bernoulli or gaussian")
    if not math.isfinite(observation_std) or observation_std <= 0:
        raise ValueError("observation_std must be finite and positive")


def prepare_observations(
    images: torch.Tensor, likelihood: str, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bernoulli uses dynamic binarization; Gaussian retains original grayscale.

    Encoder inputs and reconstruction targets are the SAME observed tensor.
    A fixed evaluation seed fixes both binarization and posterior draws.
    """
    if likelihood == "bernoulli":
        return torch.bernoulli(images, generator=generator)
    if likelihood == "gaussian":
        return images
    raise ValueError("unknown likelihood")


def reconstruction_nll(
    logits: torch.Tensor, targets: torch.Tensor, *, likelihood: str = "bernoulli",
    observation_std: float = 0.1,
) -> torch.Tensor:
    """Negative log p(x|z), summed over 784 pixels, nats per example.

    Bernoulli targets are binary (never call soft-target BCE an exact image
    likelihood). Gaussian is a continuous density on R^784 with a sigmoid mean;
    its normalization constant is INCLUDED and values can be negative. It is
    neither a truncated Gaussian nor an 8-bit discrete image likelihood.
    """
    validate_likelihood(likelihood, observation_std)
    if likelihood == "bernoulli":
        if torch.any((targets != 0) & (targets != 1)):
            raise ValueError("Bernoulli likelihood requires binary observations")
        values = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    else:
        values = 0.5 * ((targets - logits.sigmoid()) / observation_std).square()
        values = values + math.log(observation_std) + 0.5 * math.log(2 * math.pi)
    return values.flatten(1).sum(1)


def vae_terms(
    output: VAEOutput, targets: torch.Tensor, *, has_flow: bool, beta: float = 1.0,
    likelihood: str = "bernoulli", observation_std: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Per-example terms; optimize their batch mean.

    Ordinary VAE uses exact Gaussian KL. f-VAE uses a one-sample estimate
    log qK - log p, which may be negative on an individual example/batch;
    clamping it would bias the objective. `negative_elbo` always uses beta=1.
    With beta!=1, `objective` is a beta-VAE objective, not the ordinary ELBO.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    reconstruction = reconstruction_nll(
        output.logits, targets, likelihood=likelihood, observation_std=observation_std,
    )
    kl = (output.log_q - output.log_prior if has_flow
          else standard_normal_kl(output.mean, output.log_variance))
    return {"objective": reconstruction + beta * kl, "negative_elbo": reconstruction + kl,
            "reconstruction_nll": reconstruction, "kl": kl}
