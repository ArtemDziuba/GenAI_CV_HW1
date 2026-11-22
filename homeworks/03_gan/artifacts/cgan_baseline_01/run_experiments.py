"""User-authorized Digits/Fashion cGAN baselines and labeled 10x10 grids.

No model changes: reuse gan.train, its frozen evaluation bundles, and previews.
Refuse existing outputs. Run two short CUDA checks before the two full runs.
"""

import csv
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
MANIFEST = ROOT / "manifest.json"
DATASETS = (("mnist", "mnist"), ("fashion_mnist", "fashion_mnist_v2"))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(plan: dict) -> None:
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def protect(plan: dict) -> None:
    for name, expected in plan["protected_hashes"].items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError(f"Protected file changed: {name}")


def command(dataset: str, bundle: str, output: Path, *, smoke: bool) -> list[str]:
    args = ["-m", "gan.train", "--dataset", dataset, "--conditional", "--device", "cuda",
            "--evaluation-set", f"data/gan_evaluation/{bundle}",
            "--epochs", "1" if smoke else "60", "--batch-size", "64",
            "--generator-channels", "256", "--discriminator-channels", "64",
            "--generator-lr", "0.0001", "--discriminator-lr", "0.0002",
            "--latent-dim", "100", "--embedding-dim", "32",
            "--adam-beta1", "0.5", "--adam-beta2", "0.999",
            "--discriminator", "global", "--discriminator-steps", "1",
            "--feature-matching-weight", "0", "--samples-per-class", "10",
            "--sample-every", "5", "--seed", "42", "--sample-seed", "43",
            "--evaluation-seed", "44", "--output", output.as_posix()]
    if smoke:
        args += ["--train-limit", "128", "--validation-limit", "64"]
    return args


def verify(directory: Path, dataset: str, *, smoke: bool) -> dict:
    config, history, summary = (read(directory / name) for name in ("config.json", "history.json", "summary.json"))
    args = config["arguments"]
    epochs, steps = (1, 2) if smoke else (60, 60 * 844)
    assert [row["epoch"] for row in history] == list(range(1, epochs + 1))
    assert summary["last_epoch"] == epochs and summary["g_updates"] == summary["d_updates"] == steps
    assert args["dataset"] == dataset and args["conditional"] and args["resume"] is None
    assert args["generator_channels"] == 256 and args["discriminator_channels"] == 64
    assert args["generator_lr"] == .0001 and args["discriminator_lr"] == .0002
    assert args["batch_size"] == 64 and args["samples_per_class"] == 10
    if not smoke:
        assert config["split_sizes"] == {"train": 54000, "validation": 6000, "test": 10000}
    preview = torch.load(directory / "preview_inputs.pt", map_location="cpu", weights_only=True)
    assert preview["noise"].shape == (100, 100)
    assert torch.equal(preview["labels"], torch.arange(10).repeat_interleave(10))
    inputs = torch.load(directory / "evaluation_inputs.pt", map_location="cpu", weights_only=True)
    assert inputs["noise"].shape == (5000, 100) and inputs["labels"].shape == (5000,)
    for row in history:
        metrics = row["evaluation"]
        assert metrics["samples"] == metrics["real_samples"] == 5000
        assert math.isfinite(metrics["kid"]) and 0 <= metrics["conditional_accuracy"] <= 1
        assert all(math.isfinite(value) for split in ("train", "validation") for value in row[split].values())
    selected = min(history, key=lambda row: row["evaluation"]["kid"])
    assert summary["best_epoch"] == selected["epoch"] and summary["evaluation"] == selected["evaluation"]
    checkpoint = torch.load(directory / "best_kid.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == selected["epoch"]
    assert checkpoint["selection"]["score"] == selected["evaluation"]["kid"]
    for key in ("generator_optimizer", "discriminator_optimizer"):
        expected = selected["epoch"] * (2 if smoke else 844)
        assert all(state["step"].item() == expected for state in checkpoint[key]["state"].values())
    expected_previews = {1, epochs, *range(5, epochs + 1, 5)}
    assert set(summary["sampled_epochs"]) == expected_previews
    assert all((directory / f"samples_epoch_{epoch:03d}.png").is_file() for epoch in expected_previews)
    return {"best_epoch": selected["epoch"], "metrics": selected["evaluation"],
            "last_metrics": history[-1]["evaluation"], "checkpoint_sha256": digest(directory / "best_kid.pt")}


def run(plan: dict, dataset: str, bundle: str, *, smoke: bool) -> None:
    protect(plan)
    output = (Path("runs/cgan_baseline_01") / f"smoke_{dataset}" if smoke else
              Path("checkpoints/gan") / f"cgan_{dataset}_{'v2_' if dataset == 'fashion_mnist' else ''}gc256_dc64_b64_glr100u_dlr200u_kid_e60")
    if (PROJECT / output).exists():
        raise FileExistsError(output)
    args = command(dataset, bundle, output, smoke=smoke)
    trial = {"dataset": dataset, "kind": "smoke" if smoke else "full", "run": output.as_posix(),
             "command": ["uv", "run", "--locked", "python", *args],
             "status": "running", "started_utc": datetime.now(UTC).isoformat()}
    plan["trials"].append(trial)
    write(plan)
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"START {trial['kind']} {dataset}: {'1' if smoke else '60'} epochs, 100-image class grid", flush=True)
    begun, seen = time.perf_counter(), 0
    with (ROOT / "logs" / f"{trial['kind']}_{dataset}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, *args], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        trial["pid"] = process.pid
        write(plan)
        try:
            while True:
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        history = read(PROJECT / output / "history.json")
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue
                    if history and len(history) != seen:
                        seen = len(history)
                        last = history[-1]
                        best = min(history, key=lambda row: row["evaluation"]["kid"])
                        trial["progress"] = {"epoch": last["epoch"], "kid": last["evaluation"]["kid"],
                                             "best_kid": best["evaluation"]["kid"], "best_epoch": best["epoch"],
                                             "conditional_accuracy": last["evaluation"]["conditional_accuracy"]}
                        write(plan)
                        print(f"PROGRESS {dataset} epoch {last['epoch']}: KID {last['evaluation']['kid']:.4f}; "
                              f"label agreement {last['evaluation']['conditional_accuracy']:.1%}; "
                              f"best {best['evaluation']['kid']:.4f} at {best['epoch']}", flush=True)
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    trial.update(returncode=code, wall_seconds=time.perf_counter() - begun, finished_utc=datetime.now(UTC).isoformat())
    if code:
        trial["status"] = "failed"
        write(plan)
        raise RuntimeError(f"Training failed; inspect {trial['kind']}_{dataset}.log")
    trial.update(verify(PROJECT / output, dataset, smoke=smoke), status="complete")
    protect(plan)
    if not smoke:
        destination = ROOT / f"{dataset}_samples_10x10.png"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(PROJECT / output / "samples_best_kid.png", destination)
        trial["grid"] = destination.relative_to(PROJECT).as_posix()
        metadata = {"checkpoint": (output / "best_kid.pt").as_posix(),
                    "checkpoint_sha256": trial["checkpoint_sha256"], "epoch": trial["best_epoch"],
                    "rows": 10, "columns": 10, "count": 100,
                    "class_names": read(PROJECT / output / "config.json")["class_names"],
                    "labels": torch.arange(10).repeat_interleave(10).tolist(), "noise_seed": 43,
                    "grid_sha256": digest(destination)}
        (ROOT / f"{dataset}_samples_10x10.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write(plan)
    print(f"COMPLETE {trial['kind']} {dataset}: best KID {trial['metrics']['kid']:.4f} at epoch {trial['best_epoch']}", flush=True)


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError("Existing manifest: inspect it before attempting another run")
    protected = [PROJECT / "README.md", PROJECT / "AGENTS.md", PROJECT / "pyproject.toml", PROJECT / "uv.lock",
                 *sorted((PROJECT / "src/gan").glob("*.py")), *sorted((PROJECT / "src/shared").glob("*.py"))]
    for _, bundle in DATASETS:
        protected.extend((PROJECT / "data/gan_evaluation" / bundle).glob("*"))
    plan = {"status": "running", "started_utc": datetime.now(UTC).isoformat(), "trials": [],
            "checks": {"gan_model_training_evaluation_metrics": "54 passed in 20.13s"},
            "epoch_assumption": "60 epochs, matching latest vanilla comparison budget; fresh conditional initialization",
            "protected_hashes": {path.relative_to(PROJECT).as_posix(): digest(path) for path in protected if path.is_file()},
            "notes": ["KID-style MMD uses dataset-specific classifiers; scores across datasets are not comparable.",
                      "Conditional label agreement is a classifier diagnostic, not ground-truth image accuracy.",
                      "Preview: 10 samples per each of 10 classes; metric evaluation: 5,000 fixed labels/noise.",
                      "Seeded training uses nondeterministic CUDA; exact replay is not claimed."]}
    write(plan)
    try:
        for dataset, bundle in DATASETS:
            run(plan, dataset, bundle, smoke=True)
        for dataset, bundle in DATASETS:
            run(plan, dataset, bundle, smoke=False)
        rows = [{"dataset": trial["dataset"], "best_epoch": trial["best_epoch"],
                 "kid": trial["metrics"]["kid"], "conditional_accuracy": trial["metrics"]["conditional_accuracy"],
                 "mean_confidence": trial["metrics"]["mean_confidence"],
                 "confident_fraction": trial["metrics"]["confident_fraction"],
                 "confident_class_coverage": trial["metrics"]["confident_class_coverage"],
                 "run": trial["run"], "grid": trial["grid"]}
                for trial in plan["trials"] if trial["kind"] == "full"]
        with (ROOT / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        protect(plan)
        plan.update(status="complete", finished_utc=datetime.now(UTC).isoformat())
        write(plan)
        print("DONE: both 60-epoch cGANs and their labeled 10x10 best-checkpoint grids are complete.", flush=True)
    except BaseException as error:
        plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        write(plan)
        raise


if __name__ == "__main__":
    main()
