"""AE optimization, ELBO evaluation, and importance-weighted likelihood bounds."""

import math

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import Autoencoder, VAE
from .objectives import prepare_observations, reconstruction_nll, vae_terms, validate_likelihood


def run_epoch(
    model: Autoencoder | VAE, loader: DataLoader, device: torch.device, *,
    optimizer: torch.optim.Optimizer | None = None, likelihood: str = "bernoulli",
    observation_std: float = 0.1, beta: float = 1.0, seed: int = 42,
    grad_clip: float = 0.0, progress: bool = False, ae_loss: str = "mse",
) -> dict[str, float]:
    """Dataset means, weighted by batch size including a partial last batch.

    AE objective is SSE/image (mse) or SAE/image (l1); reported MSE/MAE
    divide these sums by 784. VAE terms are
    nats/image, summed over pixels and latents BEFORE taking the batch mean.
    Validation repeats its local RNG seed; training should change it by epoch.
    `grayscale_mse` compares observation means against original gray images;
    this is a reconstruction diagnostic, not a likelihood or training objective.
    """
    validate_likelihood(likelihood, observation_std)
    if ae_loss not in ("mse", "l1"):
        raise ValueError("ae_loss must be mse or l1")
    if not isinstance(model, Autoencoder) and ae_loss != "mse":
        raise ValueError("ae_loss is only meaningful for an autoencoder")
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(grad_clip) or grad_clip < 0:
        raise ValueError("beta and grad_clip must be finite and nonnegative")
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, float] = {}
    count = 0
    with torch.set_grad_enabled(training):
        for images, _ in tqdm(loader, disable=not progress, desc="train" if training else "eval", leave=False):
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if isinstance(model, Autoencoder):
                reconstructed = model(images)
                error = reconstructed - images
                objective = error.square() if ae_loss == "mse" else error.abs()
                terms = {"objective": objective.flatten(1).sum(1),
                         "grayscale_mae": error.abs().flatten(1).mean(1)}
            else:
                observed = prepare_observations(images, likelihood, generator=generator)
                output = model(observed, generator=generator)
                reconstructed = output.logits.sigmoid()
                terms = vae_terms(
                    output, observed, has_flow=bool(model.posterior_flow.layers), beta=beta,
                    likelihood=likelihood, observation_std=observation_std,
                )
            terms["grayscale_mse"] = (reconstructed - images).square().flatten(1).mean(1)
            loss = terms["objective"].mean()
            if not all(torch.isfinite(value).all() for value in terms.values()):
                raise FloatingPointError("nonfinite loss/metric; check learning rate, posterior scale, and inputs")
            if training:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value.detach().double().sum().item()
            count += images.shape[0]
    if not count:
        raise ValueError("loader must contain at least one image")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def importance_evaluation(
    model: VAE, loader: DataLoader, device: torch.device, *, samples: int = 32,
    likelihood: str = "bernoulli", observation_std: float = 0.1, seed: int = 42,
    progress: bool = False,
) -> dict[str, float]:
    """Estimate E[-log mean_k p(x,z_k)/q(z_k|x)] in nats/image.

    This is a Monte Carlo estimate of a negative IWAE bound, NOT exact NLL.
    All K draws for one image use the same observed/binarized image. Neither
    beta nor display temperature enters likelihood evaluation. Memory is O(KB)
    for scalar log weights, rather than K copies of decoder activations.
    """
    if samples < 1:
        raise ValueError("importance samples must be positive")
    validate_likelihood(likelihood, observation_std)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for images, _ in tqdm(loader, disable=not progress, desc="IWAE eval", leave=False):
        images = images.to(device, non_blocking=True)
        observed = prepare_observations(images, likelihood, generator=generator)
        weights = []
        for _ in range(samples):
            output = model(observed, generator=generator)
            reconstruction = reconstruction_nll(
                output.logits, observed, likelihood=likelihood, observation_std=observation_std,
            )
            weights.append((-reconstruction + output.log_prior - output.log_q).double())
        negative_bound = -(torch.logsumexp(torch.stack(weights), dim=0) - math.log(samples))
        if not torch.isfinite(negative_bound).all():
            raise FloatingPointError("nonfinite importance bound")
        total += negative_bound.sum().item()
        count += images.shape[0]
    if not count:
        raise ValueError("loader must contain at least one image")
    return {"negative_iwae_bound": total / count, "importance_samples": samples}
