"""Small CLI helpers shared by homework experiment runners."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .device import get_device


def resolve_device(name: str) -> torch.device:
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    return get_device() if name == "auto" else torch.device(name)


def limit_loader(loader: DataLoader, count: int | None, *, shuffle: bool) -> DataLoader:
    """Keep a prefix of an already seeded split, preserving worker/shuffle RNGs."""
    if count is None:
        return loader
    if not 1 <= count <= len(loader.dataset):
        raise ValueError("subset size must be positive and no larger than its split")
    return DataLoader(
        Subset(loader.dataset, range(count)), batch_size=loader.batch_size,
        shuffle=shuffle, num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        worker_init_fn=loader.worker_init_fn, generator=loader.generator, drop_last=False,
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
