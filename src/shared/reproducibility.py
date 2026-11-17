"""Random seeds for experiments and multiprocessing data loaders."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (including available CUDA devices).

    Deterministic mode requests deterministic PyTorch algorithms and can raise
    when an operation has no deterministic implementation. Some CUDA operations
    also require CUBLAS_WORKSPACE_CONFIG to be set before starting Python.
    Seeds alone do not guarantee identical results across devices or versions.
    """
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy from the worker seed already assigned by PyTorch.

    Pass this top-level function as DataLoader.worker_init_fn; it can be
    pickled by Windows multiprocessing. PyTorch seeds each worker itself.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
