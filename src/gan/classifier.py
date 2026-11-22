"""Dataset-specific evaluation classifier, independent of G and D."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class EvaluationClassifier(nn.Module):
    """Raw grayscale [0,1] -> fixed-dimensional features -> ten class logits."""

    def __init__(self, channels: int = 32, feature_dim: int = 128,
                 architecture: str = "simple", dropout: float = .25) -> None:
        super().__init__()
        if min(channels, feature_dim) < 1:
            raise ValueError("classifier dimensions must be positive")
        if architecture not in ("simple", "deep") or not 0 <= dropout < 1:
            raise ValueError("architecture must be simple/deep; dropout must be in [0,1)")
        if architecture == "simple":
            # Preserve the exact state-dict layout of existing evaluation bundles.
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(channels, 2 * channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(2 * channels * 7 * 7, feature_dim), nn.ReLU(),
            )
            self.head = nn.Linear(feature_dim, 10)
        else:
            # Six convolutions; pooling maps 28 -> 14 -> 7 -> 3.
            blocks = []
            inputs = 1
            for width in (channels, 2 * channels, 4 * channels):
                blocks.extend([
                    nn.Conv2d(inputs, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
                    nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(dropout / 2),
                ])
                inputs = width
            blocks.extend([nn.Flatten(), nn.Linear(4 * channels * 3 * 3, feature_dim), nn.ReLU()])
            self.features = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 10))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("classifier images must have shape (N,1,28,28)")
        features = self.features(images)
        return features, self.head(features)


@torch.no_grad()
def classification_diagnostics(logits: torch.Tensor, labels: torch.Tensor, *,
                               confidence_threshold: float = .8, bins: int = 15) -> dict:
    """Clean labelled-data diagnostics; confidence is not generated-image accuracy.

    ECE uses equal-width confidence bins, weighted by sample count. Precision and
    recall are None for classes with no predictions or no labelled support.
    These metrics do not calibrate the model or rescale its logits/features.
    """
    if (logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < 2 or labels.shape != (len(logits),)
            or labels.dtype != torch.long or not torch.isfinite(logits).all()
            or labels.min() < 0 or labels.max() >= logits.shape[1]
            or bins < 1 or not 0 <= confidence_threshold <= 1):
        raise ValueError("expected finite (N,C) logits and valid int64 labels, threshold and bin count")
    probabilities = logits.softmax(1)
    confidence, predicted = probabilities.max(1)
    correct = predicted.eq(labels)
    confident = confidence >= confidence_threshold
    boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)[1:-1]
    assignments = torch.bucketize(confidence, boundaries)
    ece = logits.new_zeros(())
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            gap = (correct[selected].float().mean() - confidence[selected].mean()).abs()
            ece += selected.float().mean() * gap
    classes = logits.shape[1]
    matrix = torch.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    support, predictions = matrix.sum(1), matrix.sum(0)
    return {"accuracy": correct.float().mean().item(), "cross_entropy": F.cross_entropy(logits, labels).item(),
            "mean_confidence": confidence.mean().item(), "confidence_threshold": confidence_threshold,
            "confident_fraction": confident.float().mean().item(),
            "confident_accuracy": correct[confident].float().mean().item() if confident.any() else None,
            "ece": ece.item(), "ece_bins": bins, "confusion_matrix": matrix.cpu().tolist(),
            "per_class_recall": [matrix[i, i].item() / support[i].item() if support[i] else None for i in range(classes)],
            "per_class_precision": [matrix[i, i].item() / predictions[i].item() if predictions[i] else None for i in range(classes)]}


def load_classifier(path: Path, device: torch.device) -> tuple[EvaluationClassifier, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("kind") != "evaluation_classifier" or checkpoint.get("format_version") != 1:
        raise ValueError("expected a dataset-specific evaluation classifier checkpoint")
    config = checkpoint["config"]
    if config["fit_pool"] != "GAN training split only; internal validation also within that pool":
        raise ValueError("classifier provenance must exclude GAN validation/reference images")
    model = EvaluationClassifier(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    return model, checkpoint
