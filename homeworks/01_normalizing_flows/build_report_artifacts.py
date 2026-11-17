"""Export existing experiment artifacts; optionally evaluate selected saved models.

Run from the repository root with uv run --locked python followed by this path.
No training, dataset downloads, or writes to the original runs are performed.
--evaluate-final evaluates missing final test results; matching saved results
are reused on subsequent invocations rather than resampling the test set.
"""

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOMEWORK_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = HOMEWORK_DIR / "artifacts"
FINAL_RUNS = {
    "mnist": "realnvp_mnist_w128_c8_b64_scale1_d5_e30",
    "fashion_mnist": "realnvp_fashion_w128_c8_b64_scale1_d5_e30",
}
REFERENCE_RUNS = {
    "mnist": "realnvp_mnist_w128_c8_b64_scale1_e30",
    "fashion_mnist": "realnvp_fashion_w128_c8_b64_scale1_e30",
}


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def collect_runs() -> dict[str, dict]:
    runs = {}
    checkpoints = PROJECT_ROOT / "checkpoints"
    # Support original flat runs and the student's archive layout. best_results
    # contains selected copies, so it must not duplicate the experiment records.
    config_paths = sorted(checkpoints.glob("*/config.json"))
    config_paths += sorted((checkpoints / "realnvp").glob("*/config.json"))
    for config_path in config_paths:
        root = config_path.parent
        if root.name in runs:
            raise ValueError(f"Duplicate experiment name in checkpoint directories: {root.name}")
        if not (root / "history.json").exists():
            continue
        history = read_json(root / "history.json")
        if not history:
            continue
        config = read_json(config_path)
        summary = read_json(root / "summary.json") if (root / "summary.json").exists() else {}
        best = min(history, key=lambda row: row["validation_nll"])
        if summary:
            assert summary["best_epoch"] == best["epoch"], root.name
            assert summary["validation"]["bpd"] == best["validation_bpd"], root.name
        runs[root.name] = {"root": root, "config": config, "history": history,
                           "summary": summary, "best": best}
    return runs


def evaluate_final(runs: dict[str, dict], evaluations: dict[str, dict]) -> None:
    import torch

    from flows.train import load_density_checkpoint
    from flows.training import run_epoch
    from shared import get_device, make_dataloaders, seed_everything

    device = get_device()
    for dataset, name in FINAL_RUNS.items():
        run = runs[name]
        path = run["root"] / "best.pt"
        digest = checkpoint_hash(path)
        if name in evaluations:
            if evaluations[name]["checkpoint_sha256"] != digest:
                raise ValueError(f"Saved test result belongs to a different checkpoint: {name}")
            print(f"Reusing recorded test evaluation: {name}", flush=True)
            continue
        args = run["config"]["arguments"]
        seed_everything(args["seed"], deterministic=args.get("deterministic", False))
        print(f"Evaluating {dataset} on {device}; no optimizer or training", flush=True)
        model, checkpoint = load_density_checkpoint(path, device)
        assert checkpoint["epoch"] == run["best"]["epoch"]
        # torch.save preserves the input-shape tuple; JSON represents it as a list.
        assert json.loads(json.dumps(checkpoint["config"]["model"])) == run["config"]["model"]
        data_root = Path(args["data_root"])
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        loaders = make_dataloaders(
            dataset, data_root, batch_size=args["batch_size"], seed=args["seed"],
            validation_fraction=args["validation_fraction"], num_workers=0,
            pin_memory=device.type == "cuda", download=False,
        )
        assert len(loaders.test.dataset) == 10000
        evaluation_seed = args["seed"] + 1  # Same convention as flows.train --evaluate-test.
        metrics = run_epoch(model, loaders.test, device, seed=evaluation_seed)
        assert not model.training
        assert all(parameter.grad is None for parameter in model.parameters())
        assert checkpoint_hash(path) == digest
        evaluations[name] = {
            "dataset": dataset,
            "checkpoint": path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": digest,
            "selected_epoch": checkpoint["epoch"],
            "model_config": checkpoint["config"]["model"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "validation": checkpoint["validation"],
            "test": metrics,
            "test_examples": len(loaders.test.dataset),
            "batch_size": args["batch_size"],
            "dequantization_seed": evaluation_seed,
            "dequantization_draws_per_image": 1,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "training_performed": False,
        }
        write_json(ARTIFACT_DIR / "final_evaluation.json", evaluations)
        print(f"{dataset}: test NLL {metrics['nll']:.6f}; test BPD {metrics['bpd']:.6f}", flush=True)
        del model, checkpoint, loaders


def export_runs(runs: dict[str, dict], evaluations: dict[str, dict]) -> None:
    records = []
    for name, run in runs.items():
        destination = ARTIFACT_DIR / "runs" / name
        destination.mkdir(parents=True, exist_ok=True)
        # Keep report assets portable while leaving weights and datasets ignored.
        for filename in ("config.json", "history.json", "summary.json", "losses.png", "samples_best.png"):
            source = run["root"] / filename
            if source.exists():
                shutil.copy2(source, destination / filename)
        config, best, history = run["config"], run["best"], run["history"]
        args = config["arguments"]
        test = evaluations[name]["test"] if name in evaluations else run["summary"].get("test")
        records.append({
            "run": name, "dataset": args["dataset"],
            "width": args["hidden_channels"], "couplings_per_group": args["couplings_per_scale"],
            "total_couplings": 2 * args["num_scales"] * args["couplings_per_scale"],
            "conditioner_depth": config["model"].get("conditioner_depth", 3),
            "batch_size": args["batch_size"], "learning_rate": args["learning_rate"],
            "scale_limit": args["scale_limit"], "logit_alpha": args["logit_alpha"],
            "seed": args["seed"], "epochs_recorded": len(history),
            "train_examples": config["split_sizes"]["train"],
            "validation_examples": config["split_sizes"]["validation"],
            "best_epoch": best["epoch"], "validation_nll": best["validation_nll"],
            "validation_bpd": best["validation_bpd"],
            "final_train_bpd": history[-1]["train_bpd"],
            "final_validation_bpd": history[-1]["validation_bpd"],
            "logged_epoch_minutes": sum(row["seconds"] for row in history) / 60,
            "test_nll": test["nll"] if test else None,
            "test_bpd": test["bpd"] if test else None,
            "test_metric_source": "final_evaluation.json" if name in evaluations
                                  else "original summary.json" if test else "not evaluated",
        })
    write_json(ARTIFACT_DIR / "experiment_metrics.json", records)
    with (ARTIFACT_DIR / "experiment_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported metadata and saved grids/curves for {len(records)} runs", flush=True)


def plot_depth_comparison(runs: dict[str, dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (dataset, final_name) in zip(axes, FINAL_RUNS.items(), strict=True):
        for name, depth, color in ((REFERENCE_RUNS[dataset], 3, "tab:blue"),
                                   (final_name, 5, "tab:orange")):
            history = runs[name]["history"]
            axis.plot([row["epoch"] for row in history],
                      [row["validation_bpd"] for row in history],
                      label=f"{depth} convolutions", color=color)
            best = runs[name]["best"]
            axis.scatter(best["epoch"], best["validation_bpd"], color=color, marker="D", s=30)
        axis.set(title="MNIST digits" if dataset == "mnist" else "Fashion-MNIST",
                 xlabel="Epoch", ylabel="Validation BPD (dequantization bound)")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(ARTIFACT_DIR / "conditioner_depth_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-final", action="store_true",
                        help="Evaluate the two selected checkpoints only if test results are missing")
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    evaluation_path = ARTIFACT_DIR / "final_evaluation.json"
    evaluations = read_json(evaluation_path) if evaluation_path.exists() else {}
    if args.evaluate_final:
        evaluate_final(runs, evaluations)
    export_runs(runs, evaluations)
    plot_depth_comparison(runs)
    lines = [
        "# Flow report artifacts", "",
        "Copied from existing runs; no model weights or datasets are included.", "",
        "- [Experiment metrics (CSV)](experiment_metrics.csv)",
        "- [Experiment metrics (JSON)](experiment_metrics.json)",
        "- [Conditioner-depth comparison](conditioner_depth_comparison.png)",
    ]
    if evaluation_path.exists():
        lines.append("- [Final test evaluation and checkpoint provenance](final_evaluation.json)")
    lines.extend(["", "| Run | Configuration | History | Original summary | Loss curves | Samples |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for name in runs:
        cells = [name]
        for filename, label in (("config.json", "config"), ("history.json", "history"),
                                ("summary.json", "summary"), ("losses.png", "curves"),
                                ("samples_best.png", "samples")):
            path = Path("runs") / name / filename
            cells.append(f"[{label}]({path.as_posix()})" if (ARTIFACT_DIR / path).exists() else "missing")
        lines.append("| " + " | ".join(cells) + " |")
    (ARTIFACT_DIR / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
