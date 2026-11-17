"""Matplotlib figures for comparable samples and experiment histories."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
from torchvision.utils import make_grid


def _save_figure(figure: Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")


def plot_image_grid(
    images: torch.Tensor,
    *,
    nrow: int = 8,
    max_images: int = 64,
    value_range: tuple[float, float] = (0.0, 1.0),
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot an (N, C, H, W) grayscale/RGB batch; return the figure to the caller.

    Detach and move images to CPU. Map the fixed value_range to [0, 1] and
    clip for display; do not normalize individual images. For GAN outputs in
    [-1, 1], pass value_range=(-1, 1). Close figures after use with plt.close.
    """
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("images must have shape (N, 1 or 3, H, W)")
    if images.shape[0] == 0 or min(images.shape[2:]) == 0:
        raise ValueError("images must be nonempty")
    if nrow < 1 or max_images < 1:
        raise ValueError("nrow and max_images must be positive")
    low, high = value_range
    if not low < high:
        raise ValueError("value_range must have increasing bounds")
    batch = images[:max_images].detach().to(device="cpu", dtype=torch.float32)
    batch = ((batch - low) / (high - low)).clamp(0, 1)
    grid = make_grid(batch, nrow=nrow, padding=2).permute(1, 2, 0).numpy()
    figure, axis = plt.subplots()
    axis.imshow(grid, vmin=0, vmax=1)
    axis.set_axis_off()
    if title is not None:
        axis.set_title(title)
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_metrics(
    history: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "Epoch",
    ylabel: str = "Metric",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot named metric series at steps 1..N without changing their reductions.

    Plot metrics with different units or normalization conventions separately.
    Return the figure; the caller controls display and closing.
    """
    if not history or any(len(values) == 0 for values in history.values()):
        raise ValueError("history must contain nonempty metric series")
    figure, axis = plt.subplots()
    for name, values in history.items():
        axis.plot(range(1, len(values) + 1), values, label=name)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure
