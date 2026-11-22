"""Export measured Fashion-MNIST results and saved grids, without training."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    plan = read(ROOT / "manifest.json")
    trials = [trial for trial in plan["trials"] if trial["status"] == "complete"]
    rows = []
    for trial in trials:
        config = read(PROJECT / trial["run"] / "config.json")
        history = read(PROJECT / trial["run"] / "history.json")
        metric = trial["metrics"]
        rows.append({"run": trial["run"], "gc": trial["gc"], "dc": trial["dc"], "batch_size": 64,
                     "generator_lr": .0002, "discriminator_lr": .0002, "epochs": 30,
                     "best_epoch": trial["best_epoch"], "kid": trial["kid"], "last_kid": trial["final_kid"],
                     "last5_mean_kid": float(np.mean([row["evaluation"]["kid"] for row in history[-5:]])),
                     "mean_confidence": metric["mean_confidence"], "confident_fraction": metric["confident_fraction"],
                     "confident_class_coverage": metric["confident_class_coverage"],
                     "normalized_class_entropy": metric["normalized_class_entropy"],
                     "normalized_confident_class_entropy": metric["normalized_confident_class_entropy"],
                     "class_counts": metric["class_counts"], "confident_class_counts": metric["confident_class_counts"],
                     "g_parameters": config["parameter_count"]["generator"], "d_parameters": config["parameter_count"]["discriminator"],
                     "g_updates": sum(row["train"]["g_updates"] for row in history),
                     "wall_seconds": trial["wall_seconds"]})
    (ROOT / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not rows:
        return
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = [f"G{row['gc']}/D{row['dc']}" for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for axis, key, title in zip(axes, ("kid", "confident_fraction", "normalized_class_entropy"),
                                ("Best validation KID-style MMD", "Confident samples (threshold 0.8)", "Normalized predicted-class entropy")):
        bars = axis.bar(names, [row[key] for row in rows])
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.2)
        if key != "kid":
            axis.set_ylim(0, 1.1)
    figure.savefig(ROOT / "width_comparison.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for trial in trials:
        history = read(PROJECT / trial["run"] / "history.json")
        for axis, start in zip(axes, (1, 5)):
            selected = [row for row in history if row["epoch"] >= start]
            line, = axis.plot([row["epoch"] for row in selected], [row["evaluation"]["kid"] for row in selected], label=f"G{trial['gc']}/D{trial['dc']}")
            if trial["best_epoch"] >= start:
                axis.scatter(trial["best_epoch"], trial["kid"], marker="*", color=line.get_color(), s=70)
    for axis, title in zip(axes, ("Full trajectories", "Epochs 5-30")):
        axis.set(xlabel="Epoch", ylabel="Validation KID-style MMD", title=title)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.savefig(ROOT / "kid_trajectories.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots((len(rows) + 2) // 3, 3, figsize=(13.5, 4.8 * ((len(rows) + 2) // 3)), squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, row in zip(axes.flat, rows):
        axis.imshow(plt.imread(PROJECT / row["run"] / "samples_best_kid.png"))
        axis.set_title(f"G{row['gc']}/D{row['dc']}: KID {row['kid']:.3f}, epoch {row['best_epoch']}/30", fontsize=10)
    figure.suptitle("Fashion-MNIST: selected checkpoints, identical preview noise")
    figure.savefig(ROOT / "samples_comparison.png", dpi=140)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    shares = np.array([row["class_counts"] for row in rows]) / 5000
    bottom = np.zeros(len(rows))
    class_names = read(PROJECT / rows[0]["run"] / "config.json")["class_names"]
    for label in range(10):
        axis.bar(names, shares[:, label], bottom=bottom, label=f"{label}: {class_names[label]}")
        bottom += shares[:, label]
    axis.set(ylabel="Predicted share among 5,000 generated samples", title="Class distribution at selected checkpoints")
    axis.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    figure.savefig(ROOT / "class_distribution.png", dpi=150)
    plt.close(figure)
    print(f"Exported {len(rows)} completed Fashion-MNIST trials; status {plan['status']}")


if __name__ == "__main__":
    main()
