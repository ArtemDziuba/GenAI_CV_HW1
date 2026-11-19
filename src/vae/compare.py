"""Collect actual run metrics and plot AE compression error versus latent size."""

import argparse
import csv
import json
import os
from itertools import groupby
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from shared.experiment import write_json


def collect_runs(root: Path) -> list[dict]:
    """Read only completed runs; configurations remain the source of truth."""
    rows = []
    for path in sorted(root.rglob("summary.json")):
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("kind") not in ("ae", "vae", "fvae"):
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run": path.parent.as_posix(), "model": config["kind"],
            "dataset": config["arguments"]["dataset"], "latent_dim": config["model"]["latent_dim"],
            "hidden_channels": config["model"]["hidden_channels"], "hidden_dim": config["model"]["hidden_dim"],
            "conv_depth": config["model"]["conv_depth"], "flow_steps": config["model"].get("flow_steps", 0),
            "flow_hidden_dim": config["model"].get("flow_hidden_dim"),
            "flow_hidden_layers": config["model"].get("flow_hidden_layers"),
            "scale_limit": config["model"].get("scale_limit"),
            "beta": config["objective"].get("beta"), "likelihood": config["objective"].get("likelihood", "none"),
            "ae_loss": config["objective"].get("ae_loss", "mse") if config["kind"] == "ae" else None,
            "observation_std": config["objective"].get("observation_std"),
            **{key: config["arguments"][key] for key in
               ("batch_size", "learning_rate", "epochs", "weight_decay", "grad_clip", "validation_fraction")},
            "train_size": config["split_sizes"]["train"],
            "validation_size": config["split_sizes"]["validation"],
            "seed": config["arguments"]["seed"], "parameters": config["parameter_count"],
            "deterministic": config["arguments"].get("deterministic", False),
            "best_epoch": summary["best_epoch"],
            **{f"validation_{key}": value for key, value in summary["validation"].items()},
        })
    return rows


COMPRESSION_CONTROLS = (
    "dataset", "hidden_channels", "hidden_dim", "conv_depth", "ae_loss",
    "batch_size", "learning_rate", "epochs", "weight_decay", "grad_clip",
    "validation_fraction", "train_size", "validation_size", "seed", "deterministic",
)


def compression_groups(rows: list[dict]) -> list[list[dict]]:
    """Group AEs with identical recorded controls except latent dimension.

    A run contributes the validation MSE of its selected checkpoint, not the
    final epoch. Separate objectives, seeds, splits and backbones stay separate.
    """
    def key(row: dict) -> str:
        controls = {name: row.get(name) for name in COMPRESSION_CONTROLS}
        controls["ae_loss"] = row.get("ae_loss", "mse")
        controls["deterministic"] = row.get("deterministic", False)
        return json.dumps(controls, sort_keys=True)

    ae_rows = sorted((row for row in rows if row["model"] == "ae"), key=key)
    return [sorted(group, key=lambda row: (row["latent_dim"], row["run"]))
            for _, group in groupby(ae_rows, key=key)]


def best_compression_runs(rows: list[dict]) -> list[dict]:
    """Select the lowest saved validation MSE per dataset and latent size.

    These points may use different backbones/objectives: this is a descriptive
    best-of-existing-runs comparison, not a controlled latent-size experiment.
    """
    best = {}
    for row in rows:
        if row["model"] != "ae":
            continue
        key = (row["dataset"], row["latent_dim"])
        if key not in best or row["validation_grayscale_mse"] < best[key]["validation_grayscale_mse"]:
            best[key] = row
    return [best[key] for key in sorted(best)]


def _compression_label(row: dict) -> str:
    return (f"C{row['hidden_channels']} H{row['hidden_dim']} D{row['conv_depth']} "
            f"{row.get('ae_loss', 'mse').upper()}")


def plot_compression(rows: list[dict], path: Path, *, highlight_best: bool = True) -> None:
    """Plot measured points; connect only matching controls at different Z.

    One-point groups remain one point. Lines are visual guides between actual
    measurements, with no interpolation of missing experiments or error bars.
    """
    groups = compression_groups(rows)
    if not groups:
        return
    datasets = sorted({group[0]["dataset"] for group in groups})
    figure, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6.5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        dataset_rows = [row for group in groups if group[0]["dataset"] == dataset for row in group]
        best = best_compression_runs(dataset_rows)
        best_names = {row["run"] for row in best}
        for group in groups:
            if group[0]["dataset"] != dataset:
                continue
            # Repeated runs at the same Z remain distinct scatter points.
            distinct_sizes = len({row["latent_dim"] for row in group}) == len(group)
            axis.plot([row["latent_dim"] for row in group],
                      [row["validation_grayscale_mse"] for row in group],
                      marker="o", linestyle="--" if len(group) > 1 and distinct_sizes else "none",
                      label=_compression_label(group[0]))
            for row in group:
                # Nearby non-best runs can overlap; their own figures label every point.
                if highlight_best and row["run"] not in best_names:
                    continue
                axis.annotate(f"{row['validation_grayscale_mse']:.4g} (e{row['best_epoch']})",
                              (row["latent_dim"], row["validation_grayscale_mse"]),
                              xytext=(5, 8), textcoords="offset points", fontsize=8)
        if highlight_best:
            axis.scatter([row["latent_dim"] for row in best],
                         [row["validation_grayscale_mse"] for row in best],
                         marker="*", s=190, facecolors="none", edgecolors="black", zorder=5,
                         label="Lowest MSE at each Z across saved configurations")
        axis.set(xlabel="Latent dimension Z (input: 784 scalars)", ylabel="Best-checkpoint validation MSE / pixel",
                 title={"mnist": "MNIST Digits", "fashion_mnist": "Fashion-MNIST"}.get(dataset, dataset))
        axis.set_xticks(sorted({row["latent_dim"] for row in dataset_rows}))
        # Log scale retains visibility of good runs alongside collapsed runs.
        if all(row["validation_grayscale_mse"] > 0 for row in dataset_rows):
            axis.set_yscale("log")
        axis.margins(x=0.2, y=0.28)
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    figure.suptitle("AE reconstruction error versus latent size")
    figure.text(0.5, 0.015, "Dashed lines: same recorded backbone and training controls. "
                "Labels: MSE and selected epoch" + (" for starred points." if highlight_best else "."),
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(figure)


def export_compression(rows: list[dict], output: Path) -> None:
    """Export dataset and per-configuration plots with measured-data provenance."""
    groups = compression_groups(rows)
    if not groups:
        return
    plot_compression(rows, output / "compression.png")
    plot_compression(best_compression_runs(rows), output / "compression_best.png")
    for dataset in sorted({group[0]["dataset"] for group in groups}):
        plot_compression([row for row in rows if row["dataset"] == dataset],
                         output / f"compression_{dataset}.png")
    individual = output / "by_configuration"
    individual.mkdir(exist_ok=True)
    lines = ["# AE reconstruction error versus latent size", "",
             "Generated from existing config.json and summary.json files only. "
             "No training, inference, downloads, or checkpoint changes were performed.", "",
             "Each point is per-pixel grayscale validation MSE at the run's best checkpoint. "
             "Epoch labels identify that checkpoint; the official test set is not used. "
             "Lower is better. Positive MSE is displayed on a logarithmic axis. "
             "Overview labels identify starred best points; individual configuration figures label every point.", "",
             "![All AE results](compression.png)", "",
             "## Best saved result at each latent size", "",
             "These selections can change backbone and objective between latent sizes. "
             "They describe the best available configurations, not the isolated effect of Z.", "",
             "![Best saved result at each Z](compression_best.png)", "",
             "| Dataset | Z | Scalar ratio 784/Z | Backbone / loss | Best epoch | Validation MSE | Source run |",
             "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for row in best_compression_runs(rows):
        lines.append(f"| {row['dataset']} | {row['latent_dim']} | {784 / row['latent_dim']:.2f}:1 | "
                     f"{_compression_label(row)} | {row['best_epoch']} | "
                     f"{row['validation_grayscale_mse']:.8f} | {Path(row['run']).name} |")
    lines.extend(["", "The scalar ratio is not a file-size or bitrate measurement.", "",
                  "## Comparisons with matching recorded controls", "",
                  "Each configuration has its own figure, including configurations with only one Z. "
                  "A single point cannot establish a latent-size trend. Additional training at other "
                  "Z values would be needed for those configurations; none was launched.", ""])
    manifest = []
    for index, group in enumerate(groups, start=1):
        first = group[0]
        filename = f"by_configuration/g{index:02d}_{first['dataset']}.png"
        plot_compression(group, output / filename, highlight_best=False)
        sizes = sorted({row["latent_dim"] for row in group})
        status = "measured latent-size comparison" if len(sizes) > 1 else "single measured latent size; more runs needed for a curve"
        lines.extend([f"### G{index:02d}: {first['dataset']} / {_compression_label(first)}", "",
                      f"Z = {', '.join(map(str, sizes))}; {status}.", "",
                      f"![G{index:02d}]({filename})", ""])
        for row in group:
            source = Path(row["run"]).resolve()
            # A relative link keeps the artifact directory portable with the repo.
            relative = Path(os.path.relpath(source, output.resolve())).as_posix()
            lines.append(f"- [{source.name}]({relative}/summary.json): "
                         f"Z{row['latent_dim']}, epoch {row['best_epoch']}, "
                         f"MSE {row['validation_grayscale_mse']:.8f}; "
                         f"[reconstructions]({relative}/reconstructions_best.png).")
        lines.append("")
        manifest.append({"group": f"G{index:02d}", "controls": {key: first.get(key) for key in COMPRESSION_CONTROLS},
                         "latent_sizes": sizes, "status": status, "figure": filename,
                         "runs": [row["run"] for row in group]})
    lines.extend(["## Interpretation limits", "",
                  "- Line segments only join observed points; missing latent sizes are not estimated.",
                  "- Fixed-control groups separate MSE/L1 objectives, seeds, training settings and backbones.",
                  "- These runs do not provide independent training repeats or uncertainty estimates.",
                  "- The L1 run remains included, even though its MSE is much worse.", ""])
    write_json(output / "compression_groups.json", manifest)
    (output / "compression.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("checkpoints/vae"))
    parser.add_argument("--output", type=Path, required=True, help="Fresh comparison directory")
    parser.add_argument("--model", choices=("ae", "vae", "fvae"), help="Collect only this model family")
    args = parser.parse_args(argv)
    matplotlib.use("Agg")
    rows = collect_runs(args.runs_root)
    if args.model:
        rows = [row for row in rows if row["model"] == args.model]
    if not rows:
        parser.error("no completed AE/VAE/f-VAE runs found")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "runs.json", rows)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    export_compression(rows, args.output)
    print(f"Collected {len(rows)} completed runs: {args.output}")


if __name__ == "__main__":
    main()
