"""Verify the bounded batch and export measured two-dataset comparisons."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from shared import plot_image_grid
from shared.experiment import resolve_device
from vae.train import load_checkpoint
from run_batch import assert_settings, write_manifest


def main():
    folder = Path(__file__).resolve().parent
    root = next(path for path in folder.parents if (path / "pyproject.toml").exists())
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] in ("training_complete", "complete")
    assert len(manifest["experiments"]) == 9
    assert all(entry["status"] in ("complete", "reused") for entry in manifest["experiments"])
    assert hashlib.sha256((root / "readme.md").read_bytes()).hexdigest() == manifest["readme_sha256"]
    assert hashlib.sha256((root / manifest["reference_run"] / "best.pt").read_bytes()).hexdigest() == manifest["reference_checkpoint_sha256"]
    entries = list(manifest["experiments"])
    for context in manifest["context_mnist_beta4"]:
        run = root / context["run"]
        assert hashlib.sha256((run / "best.pt").read_bytes()).hexdigest() == context["checkpoint_sha256"]
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        entries.append({"label": run.name, "dataset": "mnist", "beta": 4,
                        "flow_steps": config["model"]["flow_steps"],
                        "flow_hidden_dim": config["model"]["flow_hidden_dim"],
                        "output": context["run"], "reused": True, "context": True})
    rows, histories = [], {}
    for entry in entries:
        run = root / entry["output"]
        if "checkpoint_sha256" in entry:
            assert hashlib.sha256((run / "best.pt").read_bytes()).hexdigest() == entry["checkpoint_sha256"]
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        assert_settings(config, manifest["reference_config"], entry["dataset"], entry["beta"],
                        entry["flow_steps"], entry["flow_hidden_dim"])
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        assert len(history) == 30 and [epoch["epoch"] for epoch in history] == list(range(1, 31))
        assert "test" not in summary and not config["arguments"]["evaluate_test"]
        for epoch in history:
            assert all(math.isfinite(value) for split in ("train", "validation") for value in epoch[split].values())
        rows.append({"dataset": entry["dataset"], "beta": entry["beta"],
                     "flow_steps": entry["flow_steps"], "flow_hidden_dim": entry["flow_hidden_dim"],
                     "run": entry["output"], "reused": entry["reused"],
                     "context": entry.get("context", False), "parameters": config["parameter_count"],
                     "best_epoch": summary["best_epoch"], **summary["validation"],
                     "training_seconds": sum(epoch["seconds"] for epoch in history)})
        histories[entry["output"]] = history
    rows.sort(key=lambda row: (row["dataset"] != "mnist", row["beta"],
                              [(4, 32), (4, 64), (2, 64)].index((row["flow_steps"], row["flow_hidden_dim"]))))
    (folder / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for dataset_index, dataset in enumerate(("mnist", "fashion_mnist")):
        for row in [row for row in rows if row["dataset"] == dataset]:
            history = histories[row["run"]]
            label = f"beta{row['beta']} F{row['flow_steps']} FH{row['flow_hidden_dim']}"
            for axis, metric in zip(axes[dataset_index], ("negative_elbo", "kl")):
                axis.plot([epoch["epoch"] for epoch in history],
                          [epoch["validation"][metric] for epoch in history], label=label)
                axis.set(xlabel="Epoch", ylabel="Nats/image", title=f"{dataset}: validation {metric}")
        axes[dataset_index, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "validation_curves.png", dpi=150)
    plt.close(figure)
    samples_folder = folder / "samples"
    samples_folder.mkdir(exist_ok=True)
    device = resolve_device("cuda")
    torch.set_num_threads(1)
    images, sample_records = {}, []
    for row in rows:
        model, checkpoint = load_checkpoint(root / row["run"] / "best.pt", device)
        output = samples_folder / Path(row["run"]).name
        output.mkdir(exist_ok=True)
        files = []
        for seed in manifest["comparison_seeds"]:
            generator = torch.Generator(device=device).manual_seed(seed)
            means = model.sample(64, generator=generator, temperature=1.0).cpu()
            images[(row["dataset"], row["beta"], row["flow_steps"], row["flow_hidden_dim"], seed)] = means
            file = output / f"seed_{seed}.png"
            figure = plot_image_grid(means, max_images=64, nrow=8, save_path=file,
                                     title=f"Prior decoder means; seed {seed}, T=1")
            plt.close(figure)
            files.append(file.relative_to(folder).as_posix())
        sample_records.append({"run": row["run"], "checkpoint_epoch": checkpoint["epoch"],
                               "checkpoint_sha256": hashlib.sha256((root / row["run"] / "best.pt").read_bytes()).hexdigest(),
                               "seeds": manifest["comparison_seeds"], "count_per_seed": 64,
                               "temperature": 1.0, "display": "decoder means; no observation noise", "files": files})
        del model
        print(f"Exported {row['dataset']} beta{row['beta']} F{row['flow_steps']} FH{row['flow_hidden_dim']}", flush=True)
    (folder / "sample_metadata.json").write_text(json.dumps(sample_records, indent=2) + "\n", encoding="utf-8")
    for dataset in ("mnist", "fashion_mnist"):
        for seed in manifest["comparison_seeds"]:
            figure, axes = plt.subplots(2, 3, figsize=(14, 10))
            for beta_index, beta in enumerate((1, 4)):
                for column, (steps, width) in enumerate(((4, 32), (4, 64), (2, 64))):
                    axis = axes[beta_index, column]
                    grid = make_grid(images[(dataset, beta, steps, width, seed)], nrow=8, padding=2)
                    axis.imshow(grid.permute(1, 2, 0).numpy())
                    axis.set_title(f"beta {beta}, F{steps}, FH{width}")
                    axis.axis("off")
            figure.suptitle(f"{dataset}: prior decoder means, seed {seed}, T=1, 64 images per model")
            figure.tight_layout()
            figure.savefig(folder / f"{dataset}_seed_{seed}.png", dpi=150)
            plt.close(figure)
    lines = ["# f-VAE dataset and beta comparison", "",
             "Requested matrix: F4/FH32, F4/FH64 and F2/FH64 on MNIST with beta1, then Fashion-MNIST with beta1 and beta4. One matching completed MNIST beta1 run was reused; eight new runs were trained sequentially. Existing MNIST beta4 runs are included as comparison context, without retraining.", "",
             "Fixed: Z32 C32 H256 D1, two hidden layers per flow conditioner, scale limit 1, Bernoulli likelihood with dynamic binarization, 30 epochs, batch 64, Adam lr 0.001, seed 42. Splits are 54,000 train / 6,000 validation per dataset. No downloads or official test evaluation were performed.", "",
             "Validation checkpoint selection minimizes ordinary negative ELBO, even for beta4. The training objective uses the specified beta. Negative ELBO is an estimated bound, not exact marginal NLL; likelihood and KL units are nats/image. Grayscale MSE is a per-pixel diagnostic against original grayscale targets, with binarized encoder inputs.", "",
             "## Measurements", "",
             "| Dataset | Beta | F | FH | Parameters | Best epoch | Negative ELBO | Recon NLL | KL | MSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['beta']} | {row['flow_steps']} | {row['flow_hidden_dim']} | {row['parameters']:,} | {row['best_epoch']} | {row['negative_elbo']:.4f} | {row['reconstruction_nll']:.4f} | {row['kl']:.4f} | {row['grayscale_mse']:.6f} |")
    lines.extend(["", "## Generation comparisons", "",
                  "Rows show beta1 and beta4; columns show F4/FH32, F4/FH64 and F2/FH64. Each panel contains 64 prior decoder means at temperature 1. The three sampling seeds are inference repeats, not independent training runs. Latent coordinate meanings can differ across trained models; corresponding cells are not guaranteed to represent the same digit/class.", ""])
    for dataset in ("mnist", "fashion_mnist"):
        for seed in manifest["comparison_seeds"]:
            lines.extend([f"![{dataset}, seed {seed}]({dataset}_seed_{seed}.png)", ""])
    lines.extend(["## Assessment", "", manifest.get("assessment", "Pending final visual review."), "",
                  "## Verification and provenance", "",
                  "The Fashion-MNIST CUDA forward smoke check passed with finite objectives before training. All recorded histories have 30 epochs and finite metrics. Configurations were checked against the frozen reference. All comparison checkpoints loaded successfully. Reused checkpoint and README hashes were verified unchanged. Model implementations and dependencies were not modified.", "",
                  "[Manifest and exact commands](manifest.json), [metrics CSV](metrics.csv), [metrics JSON](metrics.json), [validation curves](validation_curves.png), [sample metadata](sample_metadata.json). Individual training logs and inference grids are saved in this directory. Each training run retains its best checkpoint, full configuration, histories, reconstruction grids, prior samples and traversals.", "",
                  "Comparisons use one training seed per setting. Visual preferences are qualitative and no FID/classifier-based ranking is claimed. Seeds alone do not guarantee bitwise reproducibility under nondeterministic CUDA operations.", ""])
    (folder / "results.md").write_text("\n".join(lines), encoding="utf-8")
    manifest["verification"] = {"requested_configurations": 9,
                                "new_runs": sum(not entry["reused"] for entry in manifest["experiments"]),
                                "reused_requested_runs": sum(entry["reused"] for entry in manifest["experiments"]),
                                "histories_finite": True, "epochs_per_run": 30,
                                "readme_unchanged": True, "reused_checkpoints_unchanged": True,
                                "official_test_evaluated": False, "checkpoints_loaded": len(rows)}
    write_manifest(manifest)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
