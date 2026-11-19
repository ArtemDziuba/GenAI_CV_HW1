"""Reconstruction previews retain 24 distinct aligned examples, offline."""

import matplotlib.pyplot as plt
import pytest
import torch

from vae import Autoencoder
import vae.artifacts as artifacts


def test_reconstruction_grid_has_24_aligned_comparisons(monkeypatch, synthetic_dataset, tmp_path):
    model = Autoencoder(latent_dim=2, hidden_channels=2, hidden_dim=4)
    monkeypatch.setattr(model, "forward", lambda images: images * 0.5)
    captured = []
    def capture(images, **options):
        captured.append((images.clone(), options))
        return plt.figure()
    monkeypatch.setattr(artifacts, "plot_image_grid", capture)
    path = tmp_path / "reconstructions.png"
    artifacts.save_reconstructions(model, synthetic_dataset.tensors[0], path)
    comparisons, settings = captured[0]
    reconstructed, reconstruction_settings = captured[1]
    assert settings["nrow"] == reconstruction_settings["nrow"] == 8
    assert settings["max_images"] == reconstruction_settings["max_images"] == 24
    assert comparisons.shape == (24, 1, 58, 28)
    originals = synthetic_dataset.tensors[0][:24]
    torch.testing.assert_close(comparisons[:, :, :28], originals)
    torch.testing.assert_close(comparisons[:, :, -28:], originals * 0.5)
    torch.testing.assert_close(reconstructed, originals * 0.5)
    assert reconstruction_settings["save_path"] == tmp_path / "reconstructions_only.png"
    with pytest.raises(ValueError):
        artifacts.save_reconstructions(model, originals, path, count=0)
