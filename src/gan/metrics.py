"""Unbiased polynomial feature MMD and confidence-aware class diagnostics.

The KID kernel is k(x,y)=(x.T@y/d+1)^3. Calling the result standard Inception
KID additionally requires the Inception feature space; other frozen classifier
features give a KID-style MMD. The unbiased estimate may legitimately be negative.
"""

import math

import torch


@torch.no_grad()
def polynomial_mmd(real: torch.Tensor, fake: torch.Tensor, *, block_size: int = 512) -> float:
    """Full-sample unbiased MMD-squared, excluding within-set diagonals.

    Uses float64 CPU blocks to avoid an NxN GPU allocation. No random subsets,
    positive clipping, or feature normalization change the estimator.
    """
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1] or not real.shape[1]:
        raise ValueError("real/fake features must be (N,D) matrices with the same nonzero D")
    if min(len(real), len(fake)) < 2 or block_size < 1:
        raise ValueError("MMD needs at least two samples per set and a positive block size")
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        raise ValueError("features must be finite")
    real, fake = (features.detach().to(device="cpu", dtype=torch.float64) for features in (real, fake))
    dimension = real.shape[1]
    def kernel_sum(left: torch.Tensor, right: torch.Tensor, diagonal: bool = False) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for start in range(0, len(left), block_size):
            for other in range(0, len(right), block_size):
                if diagonal and other < start:
                    continue  # Within-set kernels are symmetric; visit each off-diagonal block once.
                kernel = (left[start:start + block_size] @ right[other:other + block_size].T / dimension + 1).pow(3)
                total += kernel.sum() * (2 if diagonal and start != other else 1)
                if diagonal and start == other:
                    total -= kernel.diagonal().sum()
        return total
    n, m = len(real), len(fake)
    result = (kernel_sum(real, real, True) / (n * (n - 1))
              + kernel_sum(fake, fake, True) / (m * (m - 1))
              - 2 * kernel_sum(real, fake) / (n * m))
    if not torch.isfinite(result):
        raise FloatingPointError("nonfinite polynomial MMD")
    return result.item()


def _entropy(counts: torch.Tensor) -> float | None:
    if not counts.sum():
        return None
    frequencies = counts.double() / counts.sum()
    positive = frequencies[frequencies > 0]
    return -(positive * positive.log()).sum().item()


@torch.no_grad()
def class_statistics(probabilities: torch.Tensor, *, confidence_threshold: float = .8,
                     coverage_min_count: int = 5, requested_labels: torch.Tensor | None = None) -> dict:
    """Report raw diversity AND confidence-filtered diversity, never conflate them.

    Entropies are in nats. Confident entropy is undefined (None) if no image
    passes the confidence threshold. Confidence is a classifier diagnostic,
    not proof of recognizability or calibrated semantic correctness.
    """
    if probabilities.ndim != 2 or not len(probabilities) or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a nonempty (N,classes) matrix")
    probabilities = probabilities.detach().double().cpu()
    if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(1), torch.ones(len(probabilities), dtype=torch.float64), atol=1e-5)):
        raise ValueError("class probabilities must be finite, nonnegative, and sum to one")
    if not 0 <= confidence_threshold <= 1 or coverage_min_count < 1:
        raise ValueError("confidence threshold must be in [0,1]; coverage count positive")
    confidence, predictions = probabilities.max(1)
    classes = probabilities.shape[1]
    confident = confidence >= confidence_threshold
    counts = torch.bincount(predictions, minlength=classes)
    confident_counts = torch.bincount(predictions[confident], minlength=classes)
    entropy, confident_entropy = _entropy(counts), _entropy(confident_counts)
    metrics = {"samples": len(probabilities), "class_counts": counts.tolist(),
               "confident_class_counts": confident_counts.tolist(),
               "class_coverage": int((counts > 0).sum()),
               "confident_class_coverage": int((confident_counts >= coverage_min_count).sum()),
               "coverage_min_count": coverage_min_count,
               "class_entropy": entropy, "normalized_class_entropy": entropy / math.log(classes),
               "confident_class_entropy": confident_entropy,
               "normalized_confident_class_entropy": confident_entropy / math.log(classes) if confident_entropy is not None else None,
               "confidence_threshold": confidence_threshold, "mean_confidence": confidence.mean().item(),
               "confident_fraction": confident.double().mean().item(),
               "low_confidence_fraction": (~confident).double().mean().item(),
               "mean_predictive_entropy": -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1).mean().item()}
    if requested_labels is not None:
        labels = requested_labels.detach().cpu()
        if labels.shape != predictions.shape or labels.dtype != torch.long or (labels < 0).any() or (labels >= classes).any():
            raise ValueError("requested labels must be valid int64 class indices, one per image")
        correct = predictions == labels
        metrics["conditional_accuracy"] = correct.double().mean().item()
        metrics["confident_conditional_accuracy"] = correct[confident].double().mean().item() if confident.any() else None
    return metrics
