"""MNIST utilities with explicit download and preprocessing choices."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from .reproducibility import seed_worker


def load_dataset(
    name: str,
    root: str | Path = Path("data"),
    *,
    train: bool = True,
    download: bool = False,
    transform: Callable | None = None,
) -> Dataset:
    """Load MNIST or Fashion-MNIST through torchvision.

    Names are case-insensitive; Fashion-MNIST also accepts ``fashion_mnist``.
    The default transform returns float32 images of shape (1, 28, 28) in
    [0, 1]. It applies no normalization, binarization, or dequantization.
    A custom transform replaces this default. Relative roots use the current
    working directory. Network downloads must be explicitly enabled.
    """
    dataset_classes = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
    }
    key = name.lower().replace("-", "_")
    if key not in dataset_classes:
        raise ValueError(f"Unsupported dataset {name!r}; use 'mnist' or 'fashion_mnist'")
    return dataset_classes[key](
        root=Path(root),
        train=train,
        download=download,
        transform=transforms.ToTensor() if transform is None else transform,
    )


@dataclass(frozen=True)
class DataLoaders:
    """Training/validation partitions and the untouched official test set."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def make_dataloaders(
    name: str,
    root: str | Path = Path("data"),
    *,
    batch_size: int = 128,
    validation_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = False,
    transform: Callable | None = None,
) -> DataLoaders:
    """Create seeded loaders; shuffle training only, retain all final batches.

    Validation is held out from the official training set using a local RNG.
    Its size is floor(len(training) * validation_fraction). The split is random,
    not stratified. Each loader has an independent generator for reproducible
    shuffling and worker seeds, without consuming the global PyTorch RNG.
    With Windows multiprocessing, call this from a guarded script entry point
    (``if __name__ == '__main__':``); num_workers=0 is notebook-friendly.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")

    training = load_dataset(name, root, train=True, download=download, transform=transform)
    test = load_dataset(name, root, train=False, download=download, transform=transform)
    validation_size = int(len(training) * validation_fraction)
    if not 0 < validation_size < len(training):
        raise ValueError("validation_fraction must leave nonempty training and validation sets")
    train, validation = random_split(
        training,
        [len(training) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "drop_last": False,
    }
    return DataLoaders(
        train=DataLoader(
            train, shuffle=True, generator=torch.Generator().manual_seed(seed), **options
        ),
        validation=DataLoader(
            validation, shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **options
        ),
        test=DataLoader(
            test, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), **options
        ),
    )
