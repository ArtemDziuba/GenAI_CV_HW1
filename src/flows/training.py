"""Likelihood training/evaluation helpers, independent of model definitions."""

import math

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .likelihood import RealNVPDensity, negative_log_likelihood


def run_epoch(
    model: RealNVPDensity,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int = 42,
    progress: bool = False,
) -> dict[str, float]:
    """Train if optimizer is supplied; otherwise evaluate without gradients.

    Return sample-weighted mean NLL (nats/image) and BPD, including the bin
    correction. One uniform dequantization draw per image is used. Fixed seed
    and fixed evaluation order give repeatable validation noise without changing
    the global RNG. Training should pass a different seed each epoch.
    """
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator(device=device).manual_seed(seed)
    total_nll, total_images = 0.0, 0
    batches = tqdm(loader, desc="Train" if training else "Evaluate", disable=not progress)
    with torch.set_grad_enabled(training):
        for images, _ in batches:
            images = images.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            log_prob = model.log_prob(images, generator=generator)
            loss = negative_log_likelihood(log_prob)
            if not torch.isfinite(log_prob).all():
                raise FloatingPointError("Nonfinite log likelihood encountered")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_nll += -log_prob.detach().double().sum().item()
            total_images += images.shape[0]
    if total_images == 0:
        raise ValueError("cannot run an epoch on an empty loader")
    nll = total_nll / total_images
    return {"nll": nll, "bpd": nll / (model.flow.dimensions * math.log(2))}
