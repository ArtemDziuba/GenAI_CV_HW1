"""Export measured sweep tables and plots; no model evaluation or training."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def record(run: str, role: str, status: str = "complete") -> dict:
    directory = PROJECT_ROOT / run
    config, history = read(directory / "config.json"), read(directory / "history.json")
    best = min(history, key=lambda row: row["evaluation"]["kid"])
    evaluation = best["evaluation"]
    args = config["arguments"]
    return {"run": run, "role": role, "status": status, "gc": config["generator"]["channels"],
            "dc": config["discriminator"]["channels"], "batch": args["batch_size"],
            "g_lr": args["generator_lr"], "d_lr": args["discriminator_lr"], "epochs": len(history),
            "best_epoch": best["epoch"], "kid": evaluation["kid"], "last_kid": history[-1]["evaluation"]["kid"],
            "first30_best_kid": min(row["evaluation"]["kid"] for row in history[:30]),
            "first60_best_kid": min(row["evaluation"]["kid"] for row in history[:60]),
            "last10_mean_kid": float(np.mean([row["evaluation"]["kid"] for row in history[-10:]])),
            "mean_confidence": evaluation["mean_confidence"], "confident_fraction": evaluation["confident_fraction"],
            "confident_coverage": evaluation["confident_class_coverage"],
            "normalized_class_entropy": evaluation["normalized_class_entropy"],
            "normalized_confident_class_entropy": evaluation["normalized_confident_class_entropy"],
            "class_counts": evaluation["class_counts"], "confident_class_counts": evaluation["confident_class_counts"],
            "predicted_sixes": evaluation["class_counts"][6],
            "generator_parameters": config["parameter_count"]["generator"],
            "discriminator_parameters": config["parameter_count"]["discriminator"],
            "g_updates": sum(row["train"]["g_updates"] for row in history),
            "d_updates": sum(row["train"]["d_updates"] for row in history),
            "updates_at_best": sum(row["train"]["g_updates"] for row in history if row["epoch"] <= best["epoch"]),
            "logged_seconds": sum(row["seconds"] for row in history)}


def sample_comparison(rows: list[dict], filename: str, title: str) -> None:
    """Compare saved scientific sample grids without generating new images."""
    if not rows:
        return
    columns = min(3, len(rows))
    figure, axes = plt.subplots((len(rows) + columns - 1) // columns, columns,
                               figsize=(columns * 4.5, ((len(rows) + columns - 1) // columns) * 4.8),
                               squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, row in zip(axes.flat, rows):
        axis.imshow(plt.imread(PROJECT_ROOT / row["run"] / "samples_best_kid.png"))
        axis.set_title(f"G{row['gc']}/D{row['dc']}, b{row['batch']}, {row['g_lr']:g}/{row['d_lr']:g}\n"
                       f"KID {row['kid']:.2f}; epoch {row['best_epoch']}/{row['epochs']}", fontsize=9)
    figure.suptitle(title)
    figure.savefig(ROOT / filename, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plan = read(ROOT / "manifest.json")
    rows = [record(plan["reference"]["run"], "existing_60_reference")]
    for trial in plan["trials"]:
        if trial["status"] == "complete":
            row = record(trial["run"], ",".join(trial["roles"]))
            row["wall_seconds"] = trial["wall_seconds"]
            rows.append(row)
    (ROOT / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (ROOT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    candidates = [row for row in rows if row["epochs"] == 60 and not row["role"].startswith("lr_")]
    if len(candidates) > 1:
        sample_comparison(candidates, "batch_width_samples.png", "Selected checkpoints: same preview noise")
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        names = [f"G{row['gc']}/D{row['dc']}\nb{row['batch']}" for row in candidates]
        axes[0].bar(names, [row["kid"] for row in candidates])
        axes[0].set(ylabel="Best KID-style MMD", title="60-epoch batch/width comparisons")
        axes[1].bar(names, [100 * row["confident_fraction"] for row in candidates])
        axes[1].set(ylabel="Confident generated images (%)", title="Confidence at each selected checkpoint")
        for axis in axes:
            axis.grid(axis="y", alpha=.2)
        figure.savefig(ROOT / "batch_width_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    hundred = [row for row in rows if row["epochs"] == 100]
    if hundred:
        sample_comparison([rows[0], *hundred], "confirmation_samples.png", "60-epoch reference and 100-epoch runs")
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for row in hundred:
            history = read(PROJECT_ROOT / row["run"] / "history.json")
            label = f"G{row['gc']}/D{row['dc']} b{row['batch']}, lr {row['g_lr']:g}/{row['d_lr']:g}"
            for axis, start in zip(axes, (1, 20)):
                selected = [item for item in history if item["epoch"] >= start]
                line, = axis.plot([item["epoch"] for item in selected], [item["evaluation"]["kid"] for item in selected], label=label)
                if row["best_epoch"] >= start:
                    axis.scatter(row["best_epoch"], row["kid"], marker="*", s=100, color=line.get_color())
        for axis, title in zip(axes, ("100-epoch confirmations: full trajectories", "Epochs 20-100: detailed comparison")):
            axis.set(xlabel="Epoch", ylabel="KID-style MMD", title=title)
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        figure.savefig(ROOT / "confirmations_100.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    lr_base_decision = next((decision for decision in plan["decisions"] if decision["stage"] == "learning_rate_base_100"), None)
    rate_rows = [row for row in rows if row["role"].startswith("lr_") and row["epochs"] == 60]
    if lr_base_decision and rate_rows:
        settings = lr_base_decision["settings"]
        base = next(row for row in rows if row["epochs"] == 60 and row["gc"] == settings["gc"]
                    and row["dc"] == settings["dc"] and row["batch"] == settings["batch"]
                    and row["g_lr"] == row["d_lr"] == .0002)
        ordered_rates = sorted([base, *rate_rows], key=lambda row: (row["g_lr"], row["d_lr"]))
        sample_comparison(ordered_rates, "learning_rate_samples.png", "Learning-rate grid: same preview noise")
        rates = [.0001, .0002, .0004]
        matrix = np.full((3, 3), np.nan)
        for row in [base, *rate_rows]:
            matrix[rates.index(row["g_lr"]), rates.index(row["d_lr"])] = row["kid"]
        figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
        plot = axis.imshow(matrix, cmap="viridis_r")
        for i in range(3):
            for j in range(3):
                label = f"{matrix[i, j]:.3f}" if np.isfinite(matrix[i, j]) else "Pending"
                axis.text(j, i, label, ha="center", va="center", color="black",
                          bbox={"facecolor": "white", "alpha": .7, "edgecolor": "none"})
        axis.set_xticks(range(3), ["0.0001", "0.0002", "0.0004"])
        axis.set_yticks(range(3), ["0.0001", "0.0002", "0.0004"])
        axis.set(xlabel="Discriminator learning rate", ylabel="Generator learning rate", title="60-epoch best KID (lower is better)")
        figure.colorbar(plot, ax=axis)
        figure.savefig(ROOT / "learning_rate_grid.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    confirmations = plan.get("top_three_confirmations", [])
    if len(confirmations) == 3:
        trials_by_id = {trial["id"]: trial for trial in plan["trials"]}
        rows_by_run = {row["run"]: row for row in rows}
        sixty = [rows_by_run[trials_by_id[item["source_60"]]["run"]] for item in confirmations]
        hundred = [rows_by_run[trials_by_id[item["confirmation_100"]]["run"]] for item in confirmations]
        sample_comparison([*sixty, *hundred], "top_three_samples.png",
                          "Top three: 60-epoch runs (top) and 100-epoch confirmations (bottom)")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        positions = np.arange(3)
        names = [f"G lr {row['g_lr']:g}\nD lr {row['d_lr']:g}" for row in sixty]
        for offset, selected, label in ((-.18, sixty, "60 epochs"), (.18, hundred, "100 epochs")):
            kid_bars = axes[0].bar(positions + offset, [row["kid"] for row in selected], width=.36, label=label)
            axes[0].bar_label(kid_bars, fmt="%.2f", fontsize=8)
            axes[1].bar(positions + offset, [100 * row["confident_fraction"] for row in selected], width=.36, label=label)
        for axis in axes:
            axis.set_xticks(positions, names)
            axis.grid(axis="y", alpha=.2)
            axis.legend()
        axes[0].set(ylabel="Best validation KID-style MMD", title="Top three: fresh 60- and 100-epoch runs")
        axes[1].set(ylabel="Confident generated images (%)", title="Confidence at each selected checkpoint", ylim=(0, 100))
        figure.savefig(ROOT / "top_three_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    print(f"Exported {len(rows)} completed/reference records; sweep status {plan['status']}")


if __name__ == "__main__":
    main()
