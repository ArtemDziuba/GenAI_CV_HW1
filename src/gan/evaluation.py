"""Frozen dataset-classifier evaluation with cached reference features."""

import hashlib
import json
from pathlib import Path

import torch

from .classifier import EvaluationClassifier, load_classifier
from .metrics import class_statistics, polynomial_mmd
from .models import Generator


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bundle_id(config: dict) -> str:
    fields = {key: value for key, value in config.items() if key != "bundle_id"}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def extract_features(classifier: EvaluationClassifier, images: torch.Tensor,
                     device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1 or not len(images) or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("classifier extraction needs nonempty finite [0,1] images and a positive batch size")
    features, probabilities = [], []
    classifier.eval()
    for batch in images.split(batch_size):
        feature, logits = classifier(batch.to(device))
        features.append(feature.cpu())
        probabilities.append(logits.softmax(1).cpu())
    features, probabilities = torch.cat(features), torch.cat(probabilities)
    if not torch.isfinite(features).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError("nonfinite classifier features/probabilities")
    return features, probabilities


class GeneratorEvaluator:
    """One frozen bundle and one fixed latent bank per GAN run.

    The bundle fixes real samples, feature weights, preprocessing and data-split
    membership. Only held-out validation references are accepted for selection.
    All 5,000 fake images, including low-confidence ones, enter MMD.
    """

    def __init__(self, path: Path, device: torch.device, *, latent_dim: int, conditional: bool,
                 seed: int = 44, batch_size: int = 128, confidence_threshold: float = .8,
                 coverage_min_count: int = 5, block_size: int = 512) -> None:
        if min(latent_dim, batch_size, coverage_min_count, block_size) < 1 or not 0 <= seed < 2**32:
            raise ValueError("evaluation counts must be positive; seed must be in [0,2**32)")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence threshold must be in [0,1]")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("kind") != "classifier_kid_reference" or config.get("format_version") != 1:
            raise ValueError("expected a prepared classifier KID-style evaluation bundle")
        if config["split"] != "validation" or config["bundle_id"] != bundle_id(config):
            raise ValueError("reference must be the frozen validation bundle; metadata hash mismatch")
        for filename, digest in config["file_hashes"].items():
            if file_hash(path / filename) != digest:
                raise ValueError(f"evaluation bundle changed: {filename}")
        self.classifier, classifier_checkpoint = load_classifier(path / "classifier.pt", device)
        classifier_config = classifier_checkpoint["config"]
        if (classifier_config["dataset"] != config["dataset"] or classifier_config["split_seed"] != config["split_seed"]
                or classifier_config["validation_fraction"] != config["validation_fraction"]):
            raise ValueError("classifier/reference dataset split provenance differs")
        reference = torch.load(path / "reference.pt", map_location="cpu", weights_only=True)
        self.real_features = reference["features"]
        self.count = config["count"]
        if (len(self.real_features) != self.count or len(reference["labels"]) != self.count
                or len(reference["images"]) != self.count or self.count < 2):
            raise ValueError("reference feature/label counts disagree")
        self.noise = torch.randn(self.count, latent_dim, generator=torch.Generator().manual_seed(seed))
        self.labels = reference["labels"].clone() if conditional else None
        self.config, self.device, self.batch_size = config, device, batch_size
        self.threshold, self.coverage_min_count, self.block_size = confidence_threshold, coverage_min_count, block_size
        self.protocol = {"metric": "dataset-classifier polynomial MMD (KID-style, not Inception KID)",
                         "kernel": "(x.T@y/feature_dim + 1)^3", "estimator": "full-sample unbiased U-statistic; no clipping or random subsets",
                         "feature_normalization": "none; raw frozen pre-logit features",
                         "bundle_id": config["bundle_id"], "dataset": config["dataset"],
                         "classifier_sha256": config["file_hashes"]["classifier.pt"],
                         "reference_sha256": config["file_hashes"]["reference.pt"],
                         "feature_dim": self.real_features.shape[1], "preprocessing": classifier_config["preprocessing"],
                         "real_count": self.count, "generated_count": self.count, "noise_seed": seed,
                         "latent_dim": latent_dim, "conditional": conditional,
                         "conditional_label_policy": "fixed real-reference labels" if conditional else None,
                         "reference_class_counts": torch.bincount(reference["labels"], minlength=10).tolist(),
                         "batch_size": batch_size, "confidence_threshold": confidence_threshold,
                         "coverage_min_count": coverage_min_count, "block_size": block_size,
                         "reference_classifier_accuracy": config["classifier_reference_accuracy"]}

    def save_inputs(self, path: Path) -> None:
        torch.save({"noise": self.noise, "labels": self.labels, "protocol": self.protocol}, path)

    @torch.no_grad()
    def evaluate(self, generator: Generator) -> dict:
        if generator.latent_dim != self.noise.shape[1] or generator.conditional != (self.labels is not None):
            raise ValueError("generator does not match the frozen evaluation inputs")
        training = generator.training
        generator.eval()
        features, probabilities = [], []
        try:
            for start in range(0, self.count, self.batch_size):
                noise = self.noise[start:start + self.batch_size].to(self.device)
                labels = self.labels[start:start + self.batch_size].to(self.device) if self.labels is not None else None
                images = generator(noise, labels)
                if not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
                    raise ValueError("generator evaluation images must be finite and in [-1,1]")
                feature, logits = self.classifier(images.add(1).mul(.5))
                features.append(feature.cpu())
                probabilities.append(logits.softmax(1).cpu())
        finally:
            generator.train(training)
        fake_features, predictions = torch.cat(features), torch.cat(probabilities)
        if len(fake_features) != self.count:
            raise ValueError("generated sample count changed")
        metrics = class_statistics(predictions, confidence_threshold=self.threshold,
                                   coverage_min_count=self.coverage_min_count, requested_labels=self.labels)
        return {"kid": polynomial_mmd(self.real_features, fake_features, block_size=self.block_size),
                "real_samples": len(self.real_features), **metrics}
