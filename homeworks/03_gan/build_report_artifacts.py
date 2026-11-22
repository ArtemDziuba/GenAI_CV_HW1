"""Export saved GAN report records/figures; no training, inference or downloads."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HOMEWORK = Path(__file__).resolve().parent
PROJECT = HOMEWORK.parents[1]
ARTIFACTS = HOMEWORK / "artifacts"
RUNS = {
    "mnist_vanilla_initial": "gan_mnist_kid_e30",
    "mnist_vanilla_final": "gan_mnist_sweep01_gc256_dc64_b64_glr200u_dlr100u_kid_e100",
    "mnist_vanilla_matched_cgan": "gan_mnist_sweep01_gc256_dc64_b64_glr100u_dlr200u_kid_e60",
    "fashion_vanilla_g256": "gan_fashion_mnist_v2_gc256_dc64_glr100u_dlr200u_resume_e27_to60",
    "fashion_vanilla_g512": "gan_fashion_mnist_v2_gc512_dc64_glr100u_dlr200u_resume_e30_to60",
    "mnist_cgan": "cgan_mnist_gc256_dc64_b64_glr100u_dlr200u_kid_e60",
    "fashion_cgan": "cgan_fashion_mnist_v2_gc256_dc64_b64_glr100u_dlr200u_kid_e60",
}
for dataset, alias in (("mnist", "mnist"), ("fashion_mnist", "fashion")):
    for variant in ("patch", "fm10", "patch_fm10"):
        version = "v2_" if dataset == "fashion_mnist" else ""
        RUNS[f"{alias}_{variant}"] = f"cgan_{dataset}_{version}gc256_dc64_b64_glr100u_dlr200u_{variant}_kid_e30"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def effective_history(directory: Path) -> list[dict]:
    config = read(directory / "config.json")
    history = read(directory / "history.json")
    if "resumption" in config:
        saved = config["resumption"]["saved_epoch"]
        parent = PROJECT / config["resumption"]["checkpoint"]
        prefix = [row for row in effective_history(parent.parent) if row["epoch"] <= saved]
        history = prefix + history
    assert [row["epoch"] for row in history] == list(range(1, config["arguments"]["epochs"] + 1)), directory
    return history


def main() -> None:
    rows, records, source_hashes = [], {}, {}
    copied_files = ("config.json", "history.json", "summary.json", "best_checkpoints.json",
                    "evaluation_protocol.json", "evaluation_history.json", "evaluation.csv",
                    "losses.png", "kid.png", "samples_best_kid.png", "samples_last.png")
    for alias, name in RUNS.items():
        source = PROJECT / "checkpoints/gan" / name
        target = ARTIFACTS / "report_runs" / alias
        target.mkdir(parents=True, exist_ok=True)
        config, summary = read(source / "config.json"), read(source / "summary.json")
        history = effective_history(source)
        best = min(history, key=lambda row: row["evaluation"]["kid"])
        assert best["epoch"] == summary["best_epoch"] and best["evaluation"] == summary["evaluation"], name
        assert all(row["evaluation"]["samples"] == row["evaluation"]["real_samples"] == 5000 for row in history)
        files = list(copied_files)
        if alias in ("mnist_cgan", "fashion_cgan"):
            files += [f"samples_epoch_{epoch:03d}.png" for epoch in (1, 10, 30, 60)]
        for filename in files:
            original = source / filename
            before = digest(original)
            shutil.copyfile(original, target / filename)
            assert digest(original) == digest(target / filename) == before
            source_hashes[original.relative_to(PROJECT).as_posix()] = before
        write(target / "effective_history.json", history)
        checkpoint = source / "best_kid.pt"
        source_hashes[checkpoint.relative_to(PROJECT).as_posix()] = digest(checkpoint)
        args, metrics = config["arguments"], best["evaluation"]
        rows.append({"alias": alias, "dataset": args["dataset"], "conditional": args["conditional"],
                     "gc": args["generator_channels"], "dc": args["discriminator_channels"],
                     "batch": args["batch_size"], "g_lr": args["generator_lr"], "d_lr": args["discriminator_lr"],
                     "head": args["discriminator"], "feature_matching_weight": args["feature_matching_weight"],
                     "epoch_budget": args["epochs"], "best_epoch": best["epoch"], "kid": metrics["kid"],
                     "conditional_accuracy": metrics.get("conditional_accuracy"),
                     "mean_confidence": metrics["mean_confidence"], "confident_fraction": metrics["confident_fraction"],
                     "confident_class_coverage": metrics["confident_class_coverage"],
                     "normalized_class_entropy": metrics["normalized_class_entropy"],
                     "source_run": source.relative_to(PROJECT).as_posix(),
                     "bundle_id": config["evaluation_protocol"]["bundle_id"],
                     "checkpoint_sha256": source_hashes[checkpoint.relative_to(PROJECT).as_posix()]})
        records[alias] = {"config": config, "history": history, "best": best}
    with (ARTIFACTS / "report_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(ARTIFACTS / "report_metrics.json", rows)
    write(ARTIFACTS / "report_provenance.json", {
        "kind": "saved-record export; no training, inference or test evaluation",
        "representative_runs": len(rows), "source_hashes": source_hashes,
        "history_policy": "Resume curves join parent epochs through saved epoch with the continuation; abandoned later parent epochs excluded",
        "baseline_ablation_policy": "First-30 baseline metrics/grids remain in cgan_variants_01; copied cGAN checkpoints here are selected across 60 epochs",
    })

    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    selected = (("mnist_vanilla_final", "Digits vanilla GAN, G256/D64, 100 epochs"),
                ("fashion_vanilla_g256", "Fashion vanilla GAN, G256/D64, continued to 60"),
                ("mnist_cgan", "Digits cGAN, G256/D64, 60 epochs"),
                ("fashion_cgan", "Fashion cGAN, G256/D64, 60 epochs"))
    for axis, (alias, title) in zip(axes.flat, selected):
        record = records[alias]
        epochs = [row["epoch"] for row in record["history"]]
        for key, color in (("g_loss", "tab:blue"), ("d_loss", "tab:orange")):
            for split, style in (("train", "-"), ("validation", "--")):
                axis.plot(epochs, [row[split][key] for row in record["history"]], style,
                          color=color, label=f"{split} {key}")
        axis.axvline(record["best"]["epoch"], color="gray", linestyle=":", label="minimum KID epoch")
        resumption = record["config"].get("resumption")
        if resumption:
            axis.axvline(resumption["saved_epoch"], color="tab:red", linestyle="-.", label="resume branch boundary")
        axis.set(xlabel="Actual epoch", ylabel="Mean adversarial objective", title=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Recorded GAN losses: diagnostics of the adversarial game, not image-quality scores")
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "gan_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(18, 10))
    for row, alias in enumerate(("mnist_cgan", "fashion_cgan")):
        for column, epoch in enumerate((1, 10, 30, 60)):
            axis = axes[row, column]
            axis.imshow(plt.imread(ARTIFACTS / "report_runs" / alias / f"samples_epoch_{epoch:03d}.png"))
            axis.set_axis_off()
            axis.set_title(f"{'Digits' if row == 0 else 'Fashion'}: epoch {epoch}")
    figure.suptitle("cGAN generation over training: identical noise and labels; epoch 60 is the final state, not necessarily the selected state")
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "cgan_generation_progress.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    lines = ["# GAN report artifact index", "", "Generated from saved records by `../build_report_artifacts.py`; no training or inference.", "",
             "[Representative metrics CSV](report_metrics.csv) · [JSON](report_metrics.json) · [Source hashes](report_provenance.json)", "",
             "## Representative run snapshots", "", "Only metadata and figures are copied. Weights remain in their original checkpoint directories.", "",
             "| Alias | Dataset / conditional | Epoch budget / selected | KID | Saved records | Selected samples | Original weights |",
             "| --- | --- | ---: | ---: | --- | --- | --- |"]
    for row in rows:
        alias = row["alias"]
        base = f"report_runs/{alias}"
        lines.append(f"| {alias} | {row['dataset']} / {row['conditional']} | {row['epoch_budget']} / {row['best_epoch']} | {row['kid']:.6f} | "
                     f"[config]({base}/config.json), [history]({base}/history.json), [effective history]({base}/effective_history.json), "
                     f"[summary]({base}/summary.json), [evaluation]({base}/evaluation.csv) | "
                     f"[grid]({base}/samples_best_kid.png) | [best_kid.pt](../../../{row['source_run']}/best_kid.pt) |")
    lines += ["", "## Experiment groups", "", "| Group | Measured data | Figures / protocol |", "| --- | --- | --- |",
              "| Initial Digits width experiments | [CSV](mnist_width_comparison_01.csv) | [width plot](mnist_width_comparison_01.png) |",
              "| Digits batch, width, learning-rate and 100-epoch runs | [CSV](gan_mnist_sweep_01/metrics.csv) | [manifest](gan_mnist_sweep_01/manifest.json), [top three](gan_mnist_sweep_01/top_three_comparison.png) |",
              "| Fashion initial width sweep: original v1 feature space | [CSV](gan_fashion_width_01/metrics.csv) | [manifest](gan_fashion_width_01/manifest.json) |",
              "| Fashion saved v1-selected checkpoints rescored with v2 | [CSV](fashion_classifier_upgrade_01/metrics.csv) | [classifier comparison](fashion_classifier_upgrade_01/classifier_comparison.json) |",
              "| Fashion lower generator LR, 30 epochs | [CSV](gan_fashion_glr100_01/metrics.csv) | [manifest](gan_fashion_glr100_01/manifest.json) |",
              "| Fashion continued to total epoch 60 | [CSV](gan_fashion_resume60_01/metrics.csv) | [manifest](gan_fashion_resume60_01/manifest.json), [curves](gan_fashion_resume60_01/kid_trajectories.png) |",
              "| cGAN baseline, 60 epochs | [CSV](cgan_baseline_01/metrics.csv) | [manifest](cgan_baseline_01/manifest.json), [commands](cgan_baseline_01/commands.txt) |",
              "| Matched 30-epoch PatchGAN/FM ablations | [CSV](cgan_variants_01/metrics.csv) | [manifest](cgan_variants_01/manifest.json), [commands](cgan_variants_01/commands.txt) |",
              "", "The ablation baseline uses recorded first-30 metrics and historical grids, not the selected best-60 weights. "
              "The old best-30 baseline weights were not retained. Fashion v1 and v2 KID magnitudes are separate scales; "
              "rescoring saved v1-selected checkpoints does not reselect all old epochs under v2.", ""]
    (ARTIFACTS / "index.md").write_text("\n".join(lines), encoding="utf-8")
    assert all(digest(PROJECT / name) == expected for name, expected in source_hashes.items())
    print(f"Exported {len(rows)} representative records, copied figure/metadata snapshots, two report figures and artifact index; source files unchanged.")


if __name__ == "__main__":
    main()
