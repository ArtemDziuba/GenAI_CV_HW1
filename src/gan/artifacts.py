"""Fixed-noise sample grids and recorded GAN diagnostic curves."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from shared import plot_image_grid, plot_metrics

from .models import Generator


def fixed_inputs(model: Generator, *, count: int = 64, samples_per_class: int = 8,
                 seed: int = 43) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Create portable CPU inputs without consuming the training RNG."""
    if min(count, samples_per_class) < 1:
        raise ValueError("sample counts must be positive")
    labels = (torch.arange(model.num_classes).repeat_interleave(samples_per_class)
              if model.conditional else None)
    actual = len(labels) if labels is not None else count
    return torch.randn(actual, model.latent_dim, generator=torch.Generator().manual_seed(seed)), labels


@torch.no_grad()
def save_samples(model: Generator, noise: torch.Tensor, labels: torch.Tensor | None,
                 path: Path, *, class_names: list[str] | None = None, columns: int = 8) -> dict[str, float]:
    """Restore G mode; no BatchNorm mutation. Pixel spread is not class coverage."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        images = model(noise.to(device), labels.to(device) if labels is not None else None)
    finally:
        model.train(was_training)
    if not torch.isfinite(images).all():
        raise FloatingPointError("nonfinite generated preview")
    figure = plot_image_grid(images, nrow=columns, max_images=len(images), value_range=(-1, 1),
                             title="Fixed noise: rows show requested classes" if labels is not None else "Fixed-noise generations")
    if labels is not None:
        names = class_names or [str(index) for index in range(model.num_classes)]
        if len(names) != model.num_classes or len(labels) != columns * model.num_classes:
            plt.close(figure)
            raise ValueError("conditional preview needs one row per class")
        axis = figure.axes[0]
        axis.set_axis_on()
        axis.set_xticks([])
        axis.set_yticks([16 + 30 * index for index in range(model.num_classes)],
                       labels=[f"{index}: {name}" for index, name in enumerate(names)], fontsize=8)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    pixels = images.detach().float().cpu().add(1).div(2).flatten(1)
    return {"preview_pixel_std": pixels.std(0, unbiased=False).mean().item(),
            "preview_pairwise_l1": (torch.pdist(pixels, p=1).mean().item() / 784 if len(pixels) > 1 else 0.)}


def _save_epoch_plot(series: dict, history: list[dict], path: Path, *, ylabel: str, title: str) -> None:
    """Resumed histories start after epoch 1; keep their actual epoch labels."""
    figure = plot_metrics(series, ylabel=ylabel, title=title)
    axis = figure.axes[0]
    for line in axis.lines:
        line.set_xdata([row["epoch"] for row in history])
    axis.relim()
    axis.autoscale_view()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_history(history: list[dict], output: Path) -> None:
    for name, keys, ylabel in (
        ("losses", ("g_loss", "d_loss"), "Mean objective"),
        ("generator_terms", ("g_adversarial_loss", "g_feature_matching_loss"), "Unweighted generator terms"),
        ("discriminator_probabilities", ("d_real_probability", "d_fake_probability", "g_fake_probability"), "Mean real probability"),
        ("discriminator_accuracy", ("d_accuracy",), "Real/fake accuracy (patch-averaged for PatchGAN)"),
    ):
        series = {f"{split}_{key}": [entry[split][key] for entry in history]
                  for split in ("train", "validation") for key in keys}
        _save_epoch_plot(series, history, output / f"{name}.png", ylabel=ylabel, title=name.replace("_", " "))


def save_evaluation_history(history: list[dict], output: Path) -> None:
    """Persist generator metrics separately from adversarial training diagnostics."""
    evaluations = [{"epoch": row["epoch"], **row["evaluation"]} for row in history]
    (output / "evaluation_history.json").write_text(json.dumps(evaluations, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = sorted({key for row in evaluations for key in row})
    with (output / "evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
                         for row in evaluations)
    for name, keys, ylabel in (
        ("kid", ("kid",), "Unbiased classifier-feature MMD squared (KID-style)"),
        ("class_coverage", ("class_coverage", "confident_class_coverage"), "Predicted classes represented"),
        ("class_entropy", ("class_entropy", "confident_class_entropy"), "Predicted-class entropy (nats)"),
        ("classifier_confidence", ("mean_confidence", "confident_fraction", "low_confidence_fraction"), "Confidence / fraction"),
    ):
        _save_epoch_plot({key: [row[key] for row in evaluations] for key in keys}, evaluations,
                         output / f"{name}.png", ylabel=ylabel, title=name.replace("_", " "))
    latest = evaluations[-1]
    figure, axis = plt.subplots()
    positions = torch.arange(len(latest["class_counts"])).numpy()
    axis.bar(positions - .2, latest["class_counts"], width=.4, label="all predictions")
    axis.bar(positions + .2, latest["confident_class_counts"], width=.4, label="confident predictions")
    axis.set(xlabel="Predicted class index", ylabel="Generated images", title=f"Class distribution, epoch {latest['epoch']}")
    axis.set_xticks(positions)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "class_counts.png", bbox_inches="tight")
    plt.close(figure)
