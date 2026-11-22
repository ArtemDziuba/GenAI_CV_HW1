"""Show parent prefixes, resumed trajectories and checkpoint/sample comparisons."""

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
    complete = [trial for trial in plan["trials"] if trial["status"] == "complete"]
    rows = []
    for parent in plan["parents"]:
        metric = read(PROJECT / parent["run"] / "summary.json")["evaluation"]
        rows.append({"role": "parent_best", "gc": parent["gc"], "dc": 64, "run": parent["run"],
                     "target_epoch": 30, "best_epoch": parent["epoch"], "kid": parent["kid"],
                     "confident_fraction": metric["confident_fraction"], "mean_confidence": metric["mean_confidence"],
                     "coverage": metric["confident_class_coverage"], "entropy": metric["normalized_class_entropy"]})
    for trial in complete:
        history = read(PROJECT / trial["run"] / "history.json")
        metric = trial["metrics"]
        rows.append({"role": "continued_to60", "gc": trial["gc"], "dc": 64, "run": trial["run"],
                     "target_epoch": 60, "resume_epoch": trial["start_epoch"], "new_epochs": trial["epochs_trained"],
                     "best_epoch": trial["best_epoch"], "kid": trial["kid"], "epoch60_kid": trial["final_kid"],
                     "last10_mean_kid": float(np.mean([row["evaluation"]["kid"] for row in history[-10:]])),
                     "confident_fraction": metric["confident_fraction"], "mean_confidence": metric["mean_confidence"],
                     "coverage": metric["confident_class_coverage"], "entropy": metric["normalized_class_entropy"],
                     "class_counts": metric["class_counts"], "wall_seconds": trial["wall_seconds"]})
    (ROOT / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    if not complete:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for trial in complete:
        parent = next(parent for parent in plan["parents"] if parent["gc"] == trial["gc"])
        prefix = [row for row in read(PROJECT / parent["run"] / "history.json") if row["epoch"] <= parent["epoch"]]
        history = read(PROJECT / trial["run"] / "history.json")
        for axis, begin in zip(axes, (1, 20)):
            before = [row for row in prefix if row["epoch"] >= begin]
            after = [row for row in history if row["epoch"] >= begin]
            line, = axis.plot([row["epoch"] for row in after], [row["evaluation"]["kid"] for row in after],
                              label=f"G{trial['gc']}/D64 continued {parent['epoch'] + 1}-60")
            axis.plot([row["epoch"] for row in before], [row["evaluation"]["kid"] for row in before],
                      color=line.get_color(), linestyle="--", alpha=.7, label=f"G{trial['gc']} parent prefix")
            axis.axvline(parent["epoch"], color=line.get_color(), linestyle=":", alpha=.4)
            axis.scatter(trial["best_epoch"], trial["kid"], marker="*", s=90, color=line.get_color())
    for axis, title in zip(axes, ("Original prefix and resumed training branches", "Epochs 20-60; vertical lines mark resumption")):
        axis.set(xlabel="Total epoch", ylabel="Validation v2 KID-style MMD", title=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.savefig(ROOT / "kid_trajectories.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(9, 9.6), layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for row in rows:
        i, j = (0 if row["role"] == "parent_best" else 1), (0 if row["gc"] == 256 else 1)
        axes[i, j].imshow(plt.imread(PROJECT / row["run"] / "samples_best_kid.png"))
        axes[i, j].set_title(f"G{row['gc']}/D64: best epoch {row['best_epoch']}, v2 KID {row['kid']:.2f}", fontsize=10)
    figure.suptitle("Identical preview noise: parent best (top), best after continuation (bottom)")
    figure.savefig(ROOT / "samples_comparison.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(9, 9.6), layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for trial in complete:
        j = 0 if trial["gc"] == 256 else 1
        for i, filename, epoch, kid in ((0, "samples_best_kid.png", trial["best_epoch"], trial["kid"]),
                                       (1, "samples_epoch_060.png", 60, trial["final_kid"])):
            axes[i, j].imshow(plt.imread(PROJECT / trial["run"] / filename))
            axes[i, j].set_title(f"G{trial['gc']}/D64: epoch {epoch}, v2 KID {kid:.2f}", fontsize=10)
    figure.suptitle("Selected best checkpoint (top), actual epoch 60 (bottom)")
    figure.savefig(ROOT / "best_vs_epoch60.png", dpi=140)
    plt.close(figure)
    print(f"Exported {len(complete)} completed continuations; status {plan['status']}")


if __name__ == "__main__":
    main()
