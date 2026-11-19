"""Export saved Task 2 metrics and figures without model loading or evaluation.

Run from the repository root with uv run --locked python followed by this path.
Original checkpoint directories are read only. No datasets or weights are copied.
"""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae.compare import collect_runs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SELECTED_RUNS = (
    "ae_mnist_z64_c32_h256_d1_e30",
    "ae_fashion_mnist_z64_c32_h256_d1_e30",
    "vae_mnist_z32_c64_h256_d1_beta1_e30",
    "vae_fashion_mnist_z32_c64_h256_d1_beta1_e30",
    "vae_fashion_mnist_z32_c64_h256_d1_gaussian_std01_beta1_e30",
    "vae_fashion_mnist_z32_c64_h256_d1_gaussian_std01_beta4_e60",
    "fvae_mnist_z32_c32_h256_d1_f4_fh64_beta1_e30",
    "fvae_mnist_z32_c32_h256_d1_f4_fh64_beta4_e30",
    "fvae_fashion_mnist_z32_c32_h256_d1_f4_fh64_beta1_e30",
    "fvae_fashion_mnist_z32_c32_h256_d1_f4_fh64_beta4_e30",
)
FIGURES = (
    "reconstructions_best.png", "reconstructions_best_only.png", "samples_best.png",
    "traversal_best.png", "objective.png", "negative_elbo.png", "reconstruction_nll.png",
    "kl.png", "grayscale_mse.png",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_training_terms(rows_by_name: dict[str, dict], kind: str) -> None:
    """Show unmodified logged epoch averages for the matched dataset backbones."""
    names = [name for name in SELECTED_RUNS if name.startswith(kind + "_")
             and "gaussian" not in name and "beta4" not in name]
    figure, axes = plt.subplots(len(names), 3, figsize=(13, 3.5 * len(names)),
                                squeeze=False, layout="constrained")
    for panels, name in zip(axes, names):
        row = rows_by_name[name]
        history = json.loads((PROJECT_ROOT / row["run"] / "history.json").read_text(encoding="utf-8"))
        for axis, metric in zip(panels, ("reconstruction_nll", "kl", "negative_elbo")):
            for split in ("train", "validation"):
                axis.plot([entry["epoch"] for entry in history],
                          [entry[split][metric] for entry in history], label=split)
            axis.axvline(row["best_epoch"], color="gray", linestyle=":", label="selected epoch")
            axis.set(title=f"{row['dataset']}: {metric}", xlabel="Epoch", ylabel="Nats / image")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    title = "VAE" if kind == "vae" else "f-VAE"
    figure.suptitle(f"{title} Bernoulli beta=1: recorded training and validation terms")
    figure.savefig(ARTIFACT_DIR / f"{kind}_training_terms.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    rows = collect_runs(PROJECT_ROOT / "checkpoints" / "vae")
    rows_by_name = {}
    index = ["# Task 2 experiment artifact index", "",
             "All values are saved best-checkpoint validation measurements. "
             "The subset smoke run is marked separately. No final test evaluation is available.", "",
             "Configurations, histories and summaries are copied below; weights remain in the original run directories. "
             "Selected figure copies are included where marked. No training or model evaluation is performed.", "",
             "| Run | Scope | Best epoch | Validation MSE | Negative ELBO | Records / figures |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = Path(row["run"])
        name = source.name
        row["run"] = source.relative_to(PROJECT_ROOT).as_posix()
        row["scope"] = "full split" if (row["train_size"], row["validation_size"]) == (54000, 6000) else "subset smoke"
        history = json.loads((source / "history.json").read_text(encoding="utf-8"))
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        selected = next(entry for entry in history if entry["epoch"] == row["best_epoch"])
        metric = "grayscale_mse" if row["model"] == "ae" else "negative_elbo"
        assert selected["validation"] == summary["validation"], name
        assert selected["validation"][metric] == min(entry["validation"][metric] for entry in history), name
        assert len(history) == row["epochs"], name
        row["source_sha256"] = {}
        target = ARTIFACT_DIR / "report_runs" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "summary.json", "history.json"):
            row["source_sha256"][filename] = hashlib.sha256((source / filename).read_bytes()).hexdigest()
            shutil.copy2(source / filename, target / filename)
        if name in SELECTED_RUNS:
            for filename in FIGURES:
                if (source / filename).exists():
                    shutil.copy2(source / filename, target / filename)
        links = f"[config](report_runs/{name}/config.json), [summary](report_runs/{name}/summary.json), [history](report_runs/{name}/history.json)"
        if name in SELECTED_RUNS:
            links += f", [reconstructions](report_runs/{name}/reconstructions_best.png)"
            if row["model"] != "ae":
                links += f", [samples](report_runs/{name}/samples_best.png), [traversal](report_runs/{name}/traversal_best.png)"
        nelbo = f"{row['validation_negative_elbo']:.4f}" if "validation_negative_elbo" in row else "not applicable"
        index.append(f"| {name} | {row['scope']} | {row['best_epoch']} | {row['validation_grayscale_mse']:.8f} | {nelbo} | {links} |")
        rows_by_name[name] = row
    for name in SELECTED_RUNS:
        if name not in rows_by_name:
            raise FileNotFoundError(f"Missing report reference run: {name}")
    write_json(ARTIFACT_DIR / "experiment_metrics.json", rows)
    flat_rows = [{key: value for key, value in row.items() if key != "source_sha256"} for row in rows]
    with (ARTIFACT_DIR / "experiment_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat_rows for key in row}))
        writer.writeheader()
        writer.writerows(flat_rows)
    (ARTIFACT_DIR / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    for kind in ("vae", "fvae"):
        plot_training_terms(rows_by_name, kind)
    full = sum(row["scope"] == "full split" for row in rows)
    print(f"Exported {len(rows)} runs ({full} full-split, {len(rows) - full} subset smoke), "
          f"selected figures for {len(SELECTED_RUNS)} runs and two training-term figures. No model evaluation.")


if __name__ == "__main__":
    main()
