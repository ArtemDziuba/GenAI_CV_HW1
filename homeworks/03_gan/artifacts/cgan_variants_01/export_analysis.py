"""Compare saved 30-epoch GAN histories and grids, without additional training."""

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
LABELS = {"baseline": "cGAN baseline", "patch": "PatchGAN", "fm10": "Feature matching (10)",
          "patch_fm10": "PatchGAN + feature matching (10)"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    plan = read(ROOT / "manifest.json")
    full = [trial for trial in plan["trials"] if trial["kind"] == "full"]
    assert len(full) == 6 and all(trial["status"] == "complete" for trial in full)
    rows = []
    for baseline in plan["baselines"]:
        dataset = baseline["dataset"]
        experiments = [{**baseline, "variant": "baseline"}, *[trial for trial in full if trial["dataset"] == dataset]]
        histories = {}
        for experiment in experiments:
            variant = experiment["variant"]
            directory = PROJECT / experiment["run"]
            history = [row for row in read(directory / "history.json") if row["epoch"] <= 30]
            assert [row["epoch"] for row in history] == list(range(1, 31))
            selected = min(history, key=lambda row: row["evaluation"]["kid"])
            assert selected["epoch"] == experiment["best_epoch"] and selected["evaluation"] == experiment["metrics"]
            histories[variant] = history
            config = read(directory / "config.json")
            weight = config["arguments"]["feature_matching_weight"]
            metrics = selected["evaluation"]
            tail = history[-10:]
            adversarial = statistics.mean(row["train"]["g_adversarial_loss"] for row in tail)
            matching = statistics.mean(row["train"]["g_feature_matching_loss"] for row in tail)
            rows.append({"dataset": dataset, "variant": variant, "best_epoch": selected["epoch"],
                         "kid": metrics["kid"], "kid_change_vs_baseline_percent": 100 * (metrics["kid"] / baseline["metrics"]["kid"] - 1),
                         "epoch30_kid": history[-1]["evaluation"]["kid"],
                         "last10_mean_kid": statistics.mean(row["evaluation"]["kid"] for row in tail),
                         "last10_std_kid": statistics.pstdev(row["evaluation"]["kid"] for row in tail),
                         "conditional_accuracy": metrics["conditional_accuracy"],
                         "confident_conditional_accuracy": metrics["confident_conditional_accuracy"],
                         "mean_confidence": metrics["mean_confidence"], "confident_fraction": metrics["confident_fraction"],
                         "class_coverage": metrics["class_coverage"], "confident_class_coverage": metrics["confident_class_coverage"],
                         "normalized_class_entropy": metrics["normalized_class_entropy"],
                         "last10_train_adversarial": adversarial, "last10_train_feature_matching": matching,
                         "last10_weighted_fm_to_adversarial_ratio": weight * matching / adversarial,
                         "generator_parameters": config["parameter_count"]["generator"],
                         "discriminator_parameters": config["parameter_count"]["discriminator"],
                         "training_seconds": experiment.get("wall_seconds"), "run": experiment["run"], "grid": experiment["grid"]})
        figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        for variant, history in histories.items():
            epochs = [row["epoch"] for row in history]
            metrics = [row["evaluation"] for row in history]
            axes[0, 0].plot(epochs, [row["kid"] for row in metrics], label=LABELS[variant])
            best = min(history, key=lambda row: row["evaluation"]["kid"])
            axes[0, 0].scatter(best["epoch"], best["evaluation"]["kid"], marker="*", s=90)
            axes[0, 1].plot(epochs[9:], [row["kid"] for row in metrics[9:]], label=LABELS[variant])
            axes[1, 0].plot(epochs, [row["conditional_accuracy"] for row in metrics], label=LABELS[variant])
            axes[1, 1].plot(epochs, [row["confident_fraction"] for row in metrics], label=LABELS[variant])
        for axis, title in zip(axes.flat, ("KID-style MMD: all epochs", "KID-style MMD: epochs 10–30",
                                            "Classifier agreement with requested class", "Fraction with classifier confidence ≥0.8")):
            axis.set(xlabel="Epoch", title=title)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        for axis in axes[0]:
            # Preserve any negative unbiased estimates; retain detail near zero
            # while allowing badly failed runs and good runs on the same plot.
            axis.set_yscale("symlog", linthresh=1)
            axis.set_ylabel("MMD squared (symlog; linear within ±1)")
        figure.suptitle(f"{dataset}: equal 30-epoch budget, fixed 5,000-sample evaluation")
        figure.tight_layout()
        figure.savefig(ROOT / f"{dataset}_metrics_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        figure, axes = plt.subplots(2, 2, figsize=(14, 13))
        for axis, experiment in zip(axes.flat, experiments):
            metrics = experiment["metrics"]
            axis.imshow(mpimg.imread(PROJECT / experiment["grid"]))
            axis.set_axis_off()
            axis.set_title(f"{LABELS[experiment['variant']]} — epoch {experiment['best_epoch']}\n"
                           f"KID {metrics['kid']:.3f}; label agreement {metrics['conditional_accuracy']:.1%}")
        figure.suptitle(f"{dataset}: identical 10×10 preview inputs; best KID within epochs 1–30")
        figure.tight_layout()
        figure.savefig(ROOT / f"{dataset}_samples_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    with (ROOT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Exported eight comparison rows (two historical baselines plus six new runs), metric plots and sample panels.", flush=True)


if __name__ == "__main__":
    main()
