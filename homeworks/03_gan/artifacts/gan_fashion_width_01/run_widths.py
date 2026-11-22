"""User-authorized Fashion-MNIST width sweep, with one adaptive sixth trial.

Only generator/discriminator widths vary. The final discriminator width is
chosen from the matched G512/D128 and G512/D256 validation-KID comparison.
No model code, objectives, learning rates or completed checkpoints are changed.
"""

from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from gan.evaluation import GeneratorEvaluator


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
MANIFEST = ROOT / "manifest.json"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(plan: dict) -> None:
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def verify_frozen(plan: dict) -> None:
    for name, expected in plan["preserved_hashes"].items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError(f"Frozen source, prior result or evaluation asset changed: {name}")


def verify_trial(plan: dict, trial: dict) -> None:
    directory = PROJECT / trial["run"]
    config, history, summary = (read(directory / name) for name in ("config.json", "history.json", "summary.json"))
    assert len(history) == 30 and summary["last_epoch"] == 30
    assert config["generator"]["channels"] == trial["gc"] and config["discriminator"]["channels"] == trial["dc"]
    assert all(config["arguments"][key] == value for key, value in plan["frozen_arguments"].items())
    assert config["evaluation_protocol"] == plan["evaluation_protocol"]
    assert config["split_sizes"] == {"train": 54000, "validation": 6000, "test": 10000}
    best = min(history, key=lambda row: row["evaluation"]["kid"])
    assert summary["best_epoch"] == best["epoch"] and summary["evaluation"] == best["evaluation"]
    for row in history:
        assert row["evaluation"]["samples"] == row["evaluation"]["real_samples"] == 5000
        assert math.isfinite(row["evaluation"]["kid"])
        for split in ("train", "validation"):
            assert all(math.isfinite(value) for value in row[split].values())
    for filename, count, seed in (("evaluation_inputs.pt", 5000, 44), ("preview_inputs.pt", 64, 43)):
        inputs = torch.load(directory / filename, map_location="cpu", weights_only=True)
        expected = torch.randn(count, 100, generator=torch.Generator().manual_seed(seed))
        assert torch.equal(inputs["noise"], expected) and inputs["labels"] is None
    checkpoint = torch.load(directory / "best_kid.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == best["epoch"] and checkpoint["selection"]["score"] == best["evaluation"]["kid"]
    assert set(summary["sampled_epochs"]) == {1, 5, 10, 15, 20, 25, 30}
    assert all((directory / f"samples_epoch_{epoch:03d}.png").is_file() for epoch in summary["sampled_epochs"])
    assert (directory / "samples_best_kid.png").is_file()
    trial.update(best_epoch=best["epoch"], kid=best["evaluation"]["kid"], metrics=best["evaluation"],
                 final_kid=history[-1]["evaluation"]["kid"], checkpoint_sha256=digest(directory / "best_kid.pt"),
                 logged_seconds=sum(row["seconds"] for row in history))


def run_trial(plan: dict, gc: int, dc: int) -> dict:
    verify_frozen(plan)
    output = Path("checkpoints/gan") / f"gan_fashion_mnist_width01_gc{gc}_dc{dc}_b64_glr200u_dlr200u_kid_e30"
    if (PROJECT / output).exists():
        raise FileExistsError(output)
    arguments = []
    for key, value in plan["frozen_arguments"].items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(option)
        elif value is not None:
            arguments.extend([option, str(value)])
    arguments.extend(["--generator-channels", str(gc), "--discriminator-channels", str(dc), "--output", output.as_posix()])
    trial = {"id": f"gc{gc}_dc{dc}", "gc": gc, "dc": dc, "run": output.as_posix(), "status": "running",
             "started_utc": datetime.now(UTC).isoformat(), "command": ["uv", "run", "--locked", "python", "-m", "gan.train", *arguments]}
    plan["trials"].append(trial)
    write(plan)
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"START {trial['id']}: Fashion-MNIST, 30 epochs, batch 64, G/D lr .0002/.0002", flush=True)
    started, seen = time.perf_counter(), 0
    with (ROOT / "logs" / f"{trial['id']}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "gan.train", *arguments], cwd=PROJECT,
                                   stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        trial["pid"] = process.pid
        write(plan)
        while True:
            try:
                code = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                try:
                    history = read(PROJECT / output / "history.json")
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if len(history) != seen:
                    seen = len(history)
                    best = min(history, key=lambda row: row["evaluation"]["kid"])
                    trial["progress"] = {"epoch": seen, "kid": history[-1]["evaluation"]["kid"],
                                         "best_epoch": best["epoch"], "best_kid": best["evaluation"]["kid"]}
                    write(plan)
                    print(f"PROGRESS {trial['id']} {seen}/30: KID {history[-1]['evaluation']['kid']:.4f}, "
                          f"best {best['evaluation']['kid']:.4f} at {best['epoch']}", flush=True)
    trial.update(returncode=code, wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat())
    if code:
        trial["status"] = "failed"
        write(plan)
        raise RuntimeError(f"Training failed for {trial['id']}; inspect its log")
    verify_trial(plan, trial)
    verify_frozen(plan)
    trial["status"] = "complete"
    write(plan)
    print(f"COMPLETE {trial['id']}: best KID {trial['kid']:.6f} at epoch {trial['best_epoch']}", flush=True)
    return trial


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError("Existing sweep manifest; inspect before restarting")
    old = read(PROJECT / "homeworks/03_gan/artifacts/gan_mnist_sweep_01/manifest.json")
    bundle = Path("data/gan_evaluation/fashion_mnist")
    frozen = {**old["frozen_arguments"], "dataset": "fashion_mnist", "evaluation_set": str(bundle),
              "epochs": 30, "batch_size": 64, "generator_lr": .0002, "discriminator_lr": .0002}
    evaluator = GeneratorEvaluator(PROJECT / bundle, torch.device("cpu"), latent_dim=100, conditional=False,
                                   seed=44, batch_size=128, confidence_threshold=.8, coverage_min_count=5, block_size=512)
    paths = [*sorted((PROJECT / "src/gan").glob("*.py")), *sorted((PROJECT / "src/shared").glob("*.py")),
             *(PROJECT / name for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock")),
             *(PROJECT / bundle / name for name in ("config.json", "classifier.pt", "reference.pt")),
             *(PROJECT / "homeworks/03_gan/artifacts/gan_mnist_sweep_01" / name for name in ("manifest.json", "metrics.csv"))]
    plan = {"status": "running", "scope": "Fashion-MNIST width experiments only; six 30-epoch trials; no learning-rate trials",
            "started_utc": datetime.now(UTC).isoformat(), "frozen_arguments": frozen, "evaluation_protocol": evaluator.protocol,
            "classifier_source": "checkpoints/gan_evaluation/classifier_fashion_mnist_01",
            "classifier_preparation": "5 epochs, c32/feature128, Adam .001, batch128, seed42; fitting and internal selection confined to GAN training pool",
            "preflight": {"cuda_tests_passed": 5, "widest_g512_d512_training_batch64_and_eval_batch128": "passed"},
            "preserved_hashes": {path.relative_to(PROJECT).as_posix(): digest(path) for path in paths},
            "adaptive_rule": "Compare best validation KID at G512: D256 beats D128 -> test D512; otherwise test D64. Ties favor smaller D.",
            "trials": [], "decisions": []}
    write(plan)
    results = [run_trial(plan, gc, dc) for gc, dc in ((256, 64), (256, 128), (256, 256), (512, 256), (512, 128))]
    large, small = results[3], results[4]
    final_dc = 512 if large["kid"] < small["kid"] else 64
    plan["decisions"].append({"stage": "adaptive_discriminator", "comparison": [
        {"id": trial["id"], "kid": trial["kid"]} for trial in (large, small)], "selected_trial": f"gc512_dc{final_dc}"})
    write(plan)
    run_trial(plan, 512, final_dc)
    winner = min(plan["trials"], key=lambda trial: trial["kid"])
    plan.update(status="complete", finished_utc=datetime.now(UTC).isoformat(), best_trial=winner["id"])
    write(plan)
    from export_analysis import main as export_analysis
    export_analysis()
    print(f"SWEEP COMPLETE: best {winner['id']}, KID {winner['kid']:.6f}. No further training will start.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if MANIFEST.exists():
            plan = read(MANIFEST)
            plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
            write(plan)
        raise
