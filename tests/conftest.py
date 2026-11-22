"""Offline fixtures; plotting must not open GUI windows during tests."""

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from torch.utils.data import TensorDataset


@pytest.fixture
def synthetic_dataset() -> TensorDataset:
    """Small, distinct 28x28 images with integer labels; no network access."""
    images = torch.arange(40, dtype=torch.float32).view(-1, 1, 1, 1) / 39
    return TensorDataset(images.expand(-1, 1, 28, 28).clone(), torch.arange(40) % 10)


@pytest.fixture(scope="module")
def small_torch_thread_pool():
    """Avoid CPU thread-launch overhead for tiny convolution/Jacobian tests."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def gan_evaluation_bundle(tmp_path, synthetic_dataset):
    """Tiny synthetic-only frozen reference; never claimed to be a trained classifier."""
    import json
    from gan.classifier import EvaluationClassifier
    from gan.evaluation import bundle_id, extract_features, file_hash
    root = tmp_path / "evaluation_bundle"
    root.mkdir()
    config = {"dataset": "mnist", "model": {"channels": 2, "feature_dim": 4}, "split_seed": 42,
              "validation_fraction": .1, "preprocessing": "original grayscale [0,1]; no binarization or per-image normalization",
              "fit_pool": "GAN training split only; internal validation also within that pool"}
    classifier = EvaluationClassifier(**config["model"]).eval()
    torch.save({"kind": "evaluation_classifier", "format_version": 1, "epoch": 1, "config": config,
                "model_state": classifier.state_dict()}, root / "classifier.pt")
    images, labels = (tensor[:20] for tensor in synthetic_dataset.tensors)
    features, probabilities = extract_features(classifier, images, torch.device("cpu"), 8)
    torch.save({"images": images, "labels": labels, "features": features, "probabilities": probabilities,
                "source_indices": torch.arange(20)}, root / "reference.pt")
    metadata = {"kind": "classifier_kid_reference", "format_version": 1, "dataset": "mnist", "split": "validation",
                "split_seed": 42, "validation_fraction": .1, "count": 20, "smoke_only": True, "feature_dim": 4,
                "classifier_reference_accuracy": 0., "min_classifier_accuracy": 0.,
                "file_hashes": {name: file_hash(root / name) for name in ("classifier.pt", "reference.pt")}}
    metadata["bundle_id"] = bundle_id(metadata)
    (root / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root
