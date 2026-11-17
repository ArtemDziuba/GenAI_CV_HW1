"""Exercise torchvision dispatch and loader behavior without downloading data."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler, TensorDataset

import shared.data as data


@pytest.mark.parametrize(
    "name, class_name", [("MNIST", "MNIST"), ("Fashion-MNIST", "FashionMNIST"),
                         ("fashion_mnist", "FashionMNIST")]
)
def test_torchvision_dispatch_and_default_transform(monkeypatch, tmp_path, name, class_name):
    calls = []

    def fake_dataset(**kwargs):
        calls.append(kwargs)
        image = np.full((28, 28), 255, dtype=np.uint8)
        return TensorDataset(kwargs["transform"](image).unsqueeze(0), torch.tensor([7]))

    monkeypatch.setattr(data.datasets, class_name, fake_dataset)
    image, label = data.load_dataset(name, root=tmp_path, train=False)[0]
    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert torch.all(image == 1)
    assert label.item() == 7
    assert calls[0]["root"] == Path(tmp_path)
    assert calls[0]["train"] is False
    assert calls[0]["download"] is False


def test_explicit_download_and_custom_transform(monkeypatch, tmp_path):
    calls = []
    transform = lambda image: image
    monkeypatch.setattr(data.datasets, "MNIST", lambda **kwargs: calls.append(kwargs))
    data.load_dataset("mnist", tmp_path, download=True, transform=transform)
    assert calls[0]["download"] is True
    assert calls[0]["transform"] is transform


def test_unsupported_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        data.load_dataset("cifar10")


def test_reproducible_disjoint_loaders(monkeypatch, synthetic_dataset):
    test_dataset = TensorDataset(torch.zeros(13, 1, 28, 28), torch.arange(13) % 10)
    calls = []

    def fake_load(name, root, *, train, **kwargs):
        calls.append((train, kwargs["download"]))
        return synthetic_dataset if train else test_dataset

    monkeypatch.setattr(data, "load_dataset", fake_load)
    rng_before = torch.get_rng_state().clone()
    loaders = data.make_dataloaders("mnist", batch_size=7, validation_fraction=0.2, seed=123)
    repeated = data.make_dataloaders("mnist", batch_size=7, validation_fraction=0.2, seed=123)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert calls == [(True, False), (False, False)] * 2
    train_indices = set(loaders.train.dataset.indices)
    validation_indices = set(loaders.validation.dataset.indices)
    assert len(train_indices) == 32 and len(validation_indices) == 8
    assert train_indices.isdisjoint(validation_indices)
    assert train_indices | validation_indices == set(range(40))
    assert loaders.validation.dataset.indices == repeated.validation.dataset.indices
    assert loaders.test.dataset is test_dataset
    assert isinstance(loaders.validation.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)
    batches = list(loaders.train)
    repeated_batches = list(repeated.train)
    for (images, labels), (other_images, other_labels) in zip(batches, repeated_batches, strict=True):
        assert images.shape[1:] == (1, 28, 28)
        assert torch.isfinite(images).all()
        assert torch.equal(images, other_images)
        assert torch.equal(labels, other_labels)
    assert sum(len(labels) for _, labels in batches) == 32
    assert sum(len(labels) for _, labels in loaders.validation) == 8
    assert sum(len(labels) for _, labels in loaders.test) == 13
    changed = data.make_dataloaders("mnist", validation_fraction=0.2, seed=456)
    assert changed.validation.dataset.indices != loaders.validation.dataset.indices


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"num_workers": -1}, {"validation_fraction": 0},
    {"validation_fraction": 1}, {"seed": -1},
])
def test_invalid_loader_options(kwargs):
    with pytest.raises(ValueError):
        data.make_dataloaders("mnist", **kwargs)


def test_split_cannot_be_empty(monkeypatch, synthetic_dataset):
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: synthetic_dataset)
    with pytest.raises(ValueError, match="nonempty"):
        data.make_dataloaders("mnist", validation_fraction=0.001)
