"""Select a device without machine-specific configuration."""

import torch


def get_device() -> torch.device:
    """Prefer CUDA when available; otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
