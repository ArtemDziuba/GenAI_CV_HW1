"""Compare saved v2 rescored controls with new runs selected by v2 each epoch."""

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
    rows = [{"role": "previous_stored_checkpoint", "gc": b["gc"], "dc": 64, "run": b["run"],
             "generator_lr": .0002, "discriminator_lr": .0002, "completed_epochs": 30,
             "selected_epoch": b["stored_checkpoint_epoch"], "v2_kid": b["v2_kid"],
             "confident_fraction": b["v2_confident_fraction"], "mean_confidence": b["v2_mean_confidence"],
             "confident_coverage": b["v2_confident_coverage"], "normalized_class_entropy": b["v2_normalized_class_entropy"],
             "selection": b["selection"]} for b in plan["baselines_v2"]]
    completed = [trial for trial in plan["trials"] if trial["status"] == "complete"]
    for trial in completed:
        history = read(PROJECT / trial["run"] / "history.json")
        metric = trial["metrics"]
        rows.append({"role": "new_full_run", "gc": trial["gc"], "dc": 64, "run": trial["run"],
                     "generator_lr": .0001, "discriminator_lr": .0002, "completed_epochs": len(history),
                     "selected_epoch": trial["best_epoch"], "v2_kid": trial["kid"], "final_v2_kid": trial["final_kid"],
                     "last5_mean_kid": float(np.mean([row["evaluation"]["kid"] for row in history[-5:]])),
                     "confident_fraction": metric["confident_fraction"], "mean_confidence": metric["mean_confidence"],
                     "confident_coverage": metric["confident_class_coverage"], "normalized_class_entropy": metric["normalized_class_entropy"],
                     "class_counts": metric["class_counts"], "selection": "minimum v2 KID across all 30 epochs",
                     "g_updates": sum(row["train"]["g_updates"] for row in history), "wall_seconds": trial["wall_seconds"]})
    (ROOT / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    if not completed:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for trial in completed:
        history = read(PROJECT / trial["run"] / "history.json")
        for axis, start in zip(axes, (1, 5)):
            selected = [row for row in history if row["epoch"] >= start]
            line, = axis.plot([row["epoch"] for row in selected], [row["evaluation"]["kid"] for row in selected],
                              label=f"G{trial['gc']}/D64, G lr .0001")
            if trial["best_epoch"] >= start:
                axis.scatter(trial["best_epoch"], trial["kid"], color=line.get_color(), marker="*", s=90)
            if start == 5:
                baseline = next(row for row in rows if row["role"] == "previous_stored_checkpoint" and row["gc"] == trial["gc"])
                axis.axhline(baseline["v2_kid"], color=line.get_color(), linestyle="--", alpha=.6,
                             label=f"G{trial['gc']} previous stored checkpoint, v2 rescore")
    for axis, title in zip(axes, ("All 30 epochs: no early stopping", "Epochs 5-30; dashed controls lack full v2 histories")):
        axis.set(xlabel="Training epoch", ylabel="Validation v2 KID-style MMD", title=title)
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    figure.savefig(ROOT / "kid_trajectories.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(9, 9.6), layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for row in rows:
        i = 0 if row["role"] == "previous_stored_checkpoint" else 1
        j = 0 if row["gc"] == 256 else 1
        axis = axes[i, j]
        axis.imshow(plt.imread(PROJECT / row["run"] / "samples_best_kid.png"))
        axis.set_title(f"G{row['gc']}/D64, G lr {row['generator_lr']:g}; v2 KID {row['v2_kid']:.2f}\n"
                       f"saved epoch {row['selected_epoch']}; trained {row['completed_epochs']} epochs", fontsize=9)
    figure.suptitle("Fixed-noise samples: previous stored checkpoints (top), lower G learning rate (bottom)")
    figure.savefig(ROOT / "samples_comparison.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(9, 9.6), layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for trial in completed:
        column = 0 if trial["gc"] == 256 else 1
        directory = PROJECT / trial["run"]
        for index, filename, epoch, kid in (
            (0, "samples_best_kid.png", trial["best_epoch"], trial["kid"]),
            (1, "samples_epoch_030.png", 30, trial["final_kid"]),
        ):
            axis = axes[index, column]
            axis.imshow(plt.imread(directory / filename))
            axis.set_title(f"G{trial['gc']}/D64: epoch {epoch}, v2 KID {kid:.2f}", fontsize=10)
    figure.suptitle("All runs trained 30 epochs: selected checkpoint (top), actual epoch 30 (bottom)")
    figure.savefig(ROOT / "best_vs_epoch30.png", dpi=140)
    plt.close(figure)
    print(f"Exported {len(completed)} new runs and two stored controls; status {plan['status']}")


if __name__ == "__main__":
    main()
