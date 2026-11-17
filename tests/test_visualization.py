"""Headless plot and artifact smoke tests using synthetic inputs."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from shared import plot_image_grid, plot_metrics


def test_image_grid_saves_without_modifying_input(tmp_path):
    images = torch.linspace(-1, 1, 4 * 28 * 28).reshape(4, 1, 28, 28).requires_grad_()
    original = images.detach().clone()
    path = tmp_path / "figures" / "samples.png"
    figure = plot_image_grid(images, nrow=2, value_range=(-1, 1), save_path=path)
    try:
        assert path.stat().st_size > 0
        assert len(figure.axes) == 1
        displayed = figure.axes[0].images[0].get_array()
        assert np.isfinite(displayed).all()
        assert displayed.min() >= 0 and displayed.max() <= 1
        torch.testing.assert_close(images.detach(), original)
    finally:
        plt.close(figure)


def test_grid_uses_fixed_scale():
    figure = plot_image_grid(torch.full((1, 3, 4, 4), 0.25))
    try:
        np.testing.assert_allclose(figure.axes[0].images[0].get_array(), 0.25)
    finally:
        plt.close(figure)


def test_metric_plot_saves_and_preserves_values(tmp_path):
    history = {"train": [3.0, 2.0], "validation": [3.5, 2.5]}
    path = tmp_path / "figures" / "metrics.png"
    figure = plot_metrics(history, ylabel="Loss per image", save_path=path)
    try:
        assert path.stat().st_size > 0
        assert len(figure.axes[0].lines) == 2
        assert figure.axes[0].get_ylabel() == "Loss per image"
        for line, values in zip(figure.axes[0].lines, history.values(), strict=True):
            np.testing.assert_array_equal(line.get_xdata(), [1, 2])
            np.testing.assert_array_equal(line.get_ydata(), values)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("images", [torch.zeros(1, 28, 28), torch.zeros(0, 1, 28, 28)])
def test_invalid_grid(images):
    with pytest.raises(ValueError):
        plot_image_grid(images)


def test_empty_metric_history():
    with pytest.raises(ValueError):
        plot_metrics({})
