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
