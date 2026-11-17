"""Smoke checks for imports, RNGs, device selection, and the installed package."""

import importlib
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from shared import get_device, seed_everything, seed_worker


@pytest.mark.parametrize("name", ["shared", "flows", "vae", "gan"])
def test_package_import(name):
    assert importlib.import_module(name) is not None


def test_imports_outside_repository(tmp_path):
    # Check the editable installation, independently of pytest's sys.path.
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import shared, flows, vae, gan"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_seed_repeats_all_cpu_rngs():
    def draw():
        return random.random(), np.random.rand(4), torch.rand(4)

    seed_everything(123)
    first = draw()
    seed_everything(123)
    second = draw()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_deterministic_mode():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(42, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        seed_everything(42, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        seed_everything(42, deterministic=previous)


def test_seed_worker_uses_torch_seed(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**32 + 17)
    seed_worker(0)
    first = random.random(), np.random.random()
    seed_worker(0)
    assert first == (random.random(), np.random.random())


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_device_selection(monkeypatch, available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert get_device() == torch.device(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rng_repeats():
    seed_everything(123)
    first = torch.rand(4, device=get_device())
    seed_everything(123)
    torch.testing.assert_close(first, torch.rand(4, device=get_device()), rtol=0, atol=0)
