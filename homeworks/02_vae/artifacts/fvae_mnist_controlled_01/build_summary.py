"""Build a measured summary after all five authorized runs have completed."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    root = next(path for path in folder.parents if (path / "pyproject.toml").exists())
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] in ("training_complete", "complete")
    assert [run["label"] for run in manifest["runs"]] == ["beta4", "fh128", "fh32", "f8", "f2"]
    assert all(run["status"] == "complete" for run in manifest["runs"])
    reference = root / manifest["reference_run"]
    unchanged = hashlib.sha256((reference / "best.pt").read_bytes()).hexdigest() == manifest["reference_checkpoint_sha256"]
    assert unchanged, "The existing reference checkpoint changed"
    entries = [{"label": "reference_beta1", "output": manifest["reference_run"]}, *manifest["runs"]]
    rows, histories = [], []
    for entry in entries:
        run = root / entry["output"]
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        assert len(history) == 30
        for epoch in history:
            assert all(math.isfinite(value) for split in ("train", "validation") for value in epoch[split].values())
        for key, value in manifest["reference_config"]["model"].items():
            if key not in ("flow_steps", "flow_hidden_dim"):
                assert config["model"][key] == value, (entry["label"], key)
        for key, value in manifest["fixed_settings"].items():
            assert config["arguments"][key] == value, (entry["label"], key)
        model, objective = config["model"], config["objective"]
        rows.append({
            "experiment": entry["label"], "run": entry["output"], "beta": objective["beta"],
            "flow_steps": model["flow_steps"], "flow_hidden_dim": model["flow_hidden_dim"],
            "parameters": config["parameter_count"], "best_epoch": summary["best_epoch"],
            **summary["validation"], "training_seconds": sum(epoch["seconds"] for epoch in history),
        })
        histories.append(history)
    (folder / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row, history in zip(rows, histories):
        for axis, metric in zip(axes, ("negative_elbo", "kl")):
            axis.plot([epoch["epoch"] for epoch in history],
                      [epoch["validation"][metric] for epoch in history], label=row["experiment"])
            axis.set(xlabel="Epoch", ylabel=f"Validation {metric} (nats/image)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "validation_comparison.png", dpi=150)
    plt.close(figure)
    exports = []
    for directory in ("beta_comparison", "variant_comparison"):
        exports.extend(json.loads((folder / directory / "metadata.json").read_text(encoding="utf-8")))
    by_run = {Path(record["run"]).name: record for record in exports}
    for seed in manifest["comparison_seeds"]:
        figure, axes = plt.subplots(2, 3, figsize=(14, 10))
        for axis, row in zip(axes.flat, rows):
            record = by_run[Path(row["run"]).name]
            path = next(Path(path) for path in record["files"] if Path(path).name == f"seed_{seed}.png")
            axis.imshow(plt.imread(root / path))
            axis.set_title(f"{row['experiment']}: beta {row['beta']:g}, F{row['flow_steps']}, FH{row['flow_hidden_dim']}", fontsize=10)
            axis.axis("off")
        figure.suptitle(f"Prior decoder means: seed {seed}, temperature 1, 64 images per model")
        figure.tight_layout()
        figure.savefig(folder / f"comparison_seed_{seed}.png", dpi=150)
        plt.close(figure)
    lines = [
        "# Controlled MNIST f-VAE experiments", "",
        "Five new runs were authorized and trained sequentially, in the requested order. The existing beta1 run was reused as reference.", "",
        "Fixed settings: Z32 C32 H256 D1, Bernoulli likelihood with dynamic binarization, 30 epochs, batch 64, Adam lr 0.001, seed 42, scale limit 1, two hidden layers per flow conditioner. Splits: 54,000 train / 6,000 validation; the official test set was not evaluated.", "",
        "After choosing beta, FH128/FH32 use F4 and F8/F2 use FH64. No changes were made to model implementations, dependency versions, or README. The reference checkpoint SHA256 was verified unchanged.", "",
        "## Beta decision", "",
        f"Chosen beta: **{manifest['beta_selection']['chosen_beta']}**.", "",
        manifest["beta_selection"]["assessment"], "",
        "This choice concerns the inspected generation grids. Beta4 has a worse ordinary negative ELBO and worse reconstruction MSE than beta1; it is not a likelihood improvement.", "",
        "## Measurements", "",
        "Validation metrics are from the checkpoint minimizing ordinary negative ELBO. Likelihood/KL terms are nats/image; grayscale MSE is per pixel. The beta-weighted objective is logged separately and must not be compared directly across beta values. Negative ELBO is a Monte Carlo estimate, not exact marginal NLL.", "",
        "| Experiment | Beta | F | FH | Parameters | Best epoch | Negative ELBO | Recon NLL | KL | MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['beta']:g} | {row['flow_steps']} | {row['flow_hidden_dim']} | {row['parameters']:,} | {row['best_epoch']} | {row['negative_elbo']:.4f} | {row['reconstruction_nll']:.4f} | {row['kl']:.4f} | {row['grayscale_mse']:.6f} |")
    lines.extend(["", "## Fixed-seed comparisons", "",
                  "Each model was inspected at three inference seeds (42, 123, 2026), 64 prior draws per seed, temperature 1. Images show decoder means without observation noise. These are inference repeats, not independent training seeds; visual preferences are qualitative, not measured FID/classifier scores.", ""])
    for seed in manifest["comparison_seeds"]:
        lines.append(f"![Prior sample comparison, seed {seed}](comparison_seed_{seed}.png)")
        lines.append("")
    lines.extend(["## Assessment", "", manifest.get("qualitative_review", "Pending final visual review."), "",
                  "## Verification and files", "",
                  "Three synthetic CUDA optimizer tests passed before training. All six reference/new run histories have 30 epochs and finite recorded metrics. Fixed configurations and the reference checkpoint hash were checked. Each new run has best.pt, config.json, history.json, summary.json, loss curves, prior grids, reconstruction grids, and latent traversals.", "",
                  "Exact commands, source paths, beta decision and completion state: [manifest.json](manifest.json). Measured metrics: [CSV](metrics.csv), [JSON](metrics.json). Comparison curves: [validation_comparison.png](validation_comparison.png). The runner and inference exporter are retained in this directory with individual training logs.", "",
                  "No additional training beyond the five requested experiments was performed. Seeds alone do not guarantee bitwise reproducibility with nondeterministic CUDA operations.", ""])
    (folder / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
