"""Small GANs for 1x28x28 images; discriminator outputs are raw logits."""

import torch
from torch import nn


def _labels(labels: torch.Tensor | None, batch: int, classes: int) -> torch.Tensor:
    if labels is None or labels.shape != (batch,) or labels.dtype != torch.long:
        raise ValueError("conditional models require int64 labels with shape (batch,)")
    if labels.numel() and (labels.min() < 0 or labels.max() >= classes):
        raise ValueError("class labels are out of range")
    return labels


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)):
        nn.init.normal_(module.weight, 0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    """Noise -> 7 -> 14 -> 28, with optional concatenated class embeddings.

    Widths are 2*C then C. BatchNorm is confined to the generator; output tanh
    matches real images scaled to [-1,1]. No observation noise is added.
    """

    def __init__(self, latent_dim: int = 100, channels: int = 64, conditional: bool = False,
                 embedding_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        if min(latent_dim, channels, embedding_dim, num_classes) < 1:
            raise ValueError("model dimensions must be positive")
        self.latent_dim, self.conditional, self.num_classes = latent_dim, conditional, num_classes
        self.label_embedding = nn.Embedding(num_classes, embedding_dim) if conditional else None
        self.project = nn.Sequential(
            nn.Linear(latent_dim + (embedding_dim if conditional else 0), 2 * channels * 7 * 7),
            nn.Unflatten(1, (2 * channels, 7, 7)), nn.BatchNorm2d(2 * channels), nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2 * channels, channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1), nn.Tanh(),
        )
        self.apply(_initialize)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim or not noise.shape[0]:
            raise ValueError("noise must have shape (nonempty batch, latent_dim)")
        if self.conditional:
            labels = _labels(labels, noise.shape[0], self.num_classes)
            noise = torch.cat((noise, self.label_embedding(labels)), dim=1)
        elif labels is not None:
            raise ValueError("unconditional generator does not accept labels")
        return self.upsample(self.project(noise))


class Discriminator(nn.Module):
    """Two convolution stages with a global or 7x7 PatchGAN head.

    Patch logits have nominal receptive field 18x18 (smaller at padded edges).
    Conditional labels are embedded into a learned 28x28 input plane. The same
    penultimate feature map supports feature matching in either head variant.
    No batch normalization or dropout couples real/fake feature statistics.
    """

    def __init__(self, channels: int = 64, conditional: bool = False,
                 head: str = "global", num_classes: int = 10) -> None:
        super().__init__()
        if min(channels, num_classes) < 1 or head not in ("global", "patch"):
            raise ValueError("dimensions must be positive; head must be global or patch")
        self.conditional, self.num_classes, self.head_kind = conditional, num_classes, head
        self.label_embedding = nn.Embedding(num_classes, 28 * 28) if conditional else None
        self.features = nn.Sequential(
            nn.Conv2d(2 if conditional else 1, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 2 * channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.head = (nn.Sequential(nn.Flatten(), nn.Linear(2 * channels * 7 * 7, 1))
                     if head == "global" else nn.Conv2d(2 * channels, 1, 3, padding=1))
        self.apply(_initialize)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None, *,
                return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (batch,1,28,28)")
        if self.conditional:
            labels = _labels(labels, images.shape[0], self.num_classes)
            plane = self.label_embedding(labels).view(-1, 1, 28, 28)
            images = torch.cat((images, plane), dim=1)
        elif labels is not None:
            raise ValueError("unconditional discriminator does not accept labels")
        features = self.features(images)
        logits = self.head(features)
        return (logits, features) if return_features else logits
