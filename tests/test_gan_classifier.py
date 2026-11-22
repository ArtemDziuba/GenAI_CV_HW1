"""Classifier checkpoint compatibility, confidence diagnostics and deeper model."""

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from gan.classifier import EvaluationClassifier, classification_diagnostics, load_classifier
from gan.train_classifier import TrainingAugmentation
import gan.train_classifier as training
from shared.data import DataLoaders


pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


@pytest.mark.parametrize("architecture", ["simple", "deep"])
def test_classifier_checkpoint_roundtrip_frozen_features_and_gradients(tmp_path, architecture):
    config = {"channels": 2, "feature_dim": 4}
    if architecture == "deep":
        config.update(architecture="deep", dropout=.25)
    model = EvaluationClassifier(**config)
    images = torch.rand(4, 1, 28, 28)
    before = model.head[-1].weight.detach().clone() if architecture == "deep" else model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    features, logits = model(images)
    assert features.shape == (4, 4) and logits.shape == (4, 10)
    loss = F.cross_entropy(logits, torch.arange(4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    weight = model.head[-1].weight if architecture == "deep" else model.head.weight
    assert not torch.equal(weight, before)
    path = tmp_path / "classifier.pt"
    torch.save({"kind": "evaluation_classifier", "format_version": 1,
                "config": {"model": config, "fit_pool": "GAN training split only; internal validation also within that pool"},
                "model_state": model.state_dict()}, path)
    frozen, _ = load_classifier(path, torch.device("cpu"))
    assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())
    state = {key: value.clone() for key, value in frozen.state_dict().items()}
    first, second = frozen(images), frozen(images)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(torch.equal(value, frozen.state_dict()[key]) for key, value in state.items())


def test_confidence_diagnostics_distinguish_coverage_from_accuracy():
    logits = torch.tensor([[.9, .1], [.6, .4]]).log()
    stats = classification_diagnostics(logits, torch.tensor([0, 1]), bins=10)
    assert stats["accuracy"] == .5
    assert stats["confident_fraction"] == .5 and stats["confident_accuracy"] == 1.
    assert stats["ece"] == pytest.approx(.35)
    assert stats["confusion_matrix"] == [[1, 0], [1, 0]]
    assert stats["per_class_recall"] == [1., 0.]
    assert stats["per_class_precision"] == [.5, None]


def test_confidence_diagnostics_no_confident_samples():
    stats = classification_diagnostics(torch.zeros(3, 10), torch.arange(3))
    assert stats["confident_fraction"] == 0 and stats["confident_accuracy"] is None
    with pytest.raises(ValueError):
        classification_diagnostics(torch.full((3, 10), float("nan")), torch.arange(3))


def test_deep_training_augmentation_is_confined_to_fitting_pool(tmp_path, monkeypatch, synthetic_dataset):
    train_set = torch.utils.data.Subset(synthetic_dataset, range(30))
    reference = torch.utils.data.Subset(synthetic_dataset, range(30, 40))
    loaders = DataLoaders(DataLoader(train_set, batch_size=8), DataLoader(reference, batch_size=8), DataLoader(reference, batch_size=8))
    monkeypatch.setattr(training, "make_dataloaders", lambda *args, **kwargs: loaders)
    seen = []
    original = training.classifier_epoch
    def inspect(model, loader, device, optimizer=None):
        if optimizer is not None:
            assert isinstance(loader.dataset, TrainingAugmentation)
            subset = loader.dataset.dataset
        else:
            assert not isinstance(loader.dataset, TrainingAugmentation)
            subset = loader.dataset
        assert subset.dataset is train_set
        seen.extend(int(train_set.indices[index]) for index in subset.indices)
        return original(model, loader, device, optimizer)
    monkeypatch.setattr(training, "classifier_epoch", inspect)
    output = tmp_path / "deep"
    training.main(["--dataset", "fashion_mnist", "--output", str(output), "--device", "cpu", "--epochs", "2",
                   "--architecture", "deep", "--channels", "2", "--feature-dim", "4", "--batch-size", "8",
                   "--augment", "--optimizer", "adamw", "--weight-decay", ".0001", "--scheduler", "cosine"])
    assert set(seen) == set(range(30)) and set(seen).isdisjoint(reference.indices)
    model, checkpoint = load_classifier(output / "best.pt", torch.device("cpu"))
    assert checkpoint["config"]["model"]["architecture"] == "deep"
    assert checkpoint["validation"]["diagnostics"]["ece"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_deep_classifier_cuda_optimization():
    model = EvaluationClassifier(channels=4, feature_dim=8, architecture="deep").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    _, logits = model(torch.rand(8, 1, 28, 28, device="cuda"))
    loss = F.cross_entropy(logits, torch.arange(8, device="cuda"))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
