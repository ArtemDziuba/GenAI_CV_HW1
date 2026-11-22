"""Two full 30-epoch Fashion runs: lower G rate, fixed D rate, v2 KID."""

from datetime import UTC, datetime
import importlib.util
import json
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
# Reuse the existing 30-epoch result, noise-bank and checkpoint verification.
spec = importlib.util.spec_from_file_location("fashion_width_checks", ROOT.parent / "gan_fashion_width_01/run_widths.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
read, digest = checks.read, checks.digest


def write(plan: dict) -> None:
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def run_trial(plan: dict, gc: int) -> None:
    checks.verify_frozen(plan)
    output = Path("checkpoints/gan") / f"gan_fashion_mnist_v2_gc{gc}_dc64_b64_glr100u_dlr200u_kid_e30"
    if (PROJECT / output).exists():
        raise FileExistsError(output)
    args = []
    for key, value in plan["frozen_arguments"].items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif value is not None:
            args.extend([flag, str(value)])
    args.extend(["--generator-channels", str(gc), "--discriminator-channels", "64", "--output", output.as_posix()])
    trial = {"id": f"gc{gc}_dc64", "gc": gc, "dc": 64, "run": output.as_posix(), "status": "running",
             "started_utc": datetime.now(UTC).isoformat(), "command": ["uv", "run", "--locked", "python", "-m", "gan.train", *args]}
    plan["trials"].append(trial)
    write(plan)
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"START {trial['id']}: 30 complete epochs, G lr .0001, D lr .0002, v2 KID", flush=True)
    started, seen = time.perf_counter(), 0
    with (ROOT / "logs" / f"{trial['id']}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "gan.train", *args], cwd=PROJECT, stdout=log,
                                   stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
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
                    print(f"PROGRESS {trial['id']}: trained {seen}/30; KID {history[-1]['evaluation']['kid']:.4f}; "
                          f"best {best['evaluation']['kid']:.4f} at epoch {best['epoch']}", flush=True)
    trial.update(returncode=code, wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat())
    if code:
        trial["status"] = "failed"
        write(plan)
        raise RuntimeError(f"Training failed; inspect {trial['id']} log")
    checks.verify_trial(plan, trial)
    checks.verify_frozen(plan)
    trial.update(status="complete", completed_epochs=30)
    write(plan)
    print(f"COMPLETE {trial['id']}: all 30 epochs trained; best KID {trial['kid']:.6f} at {trial['best_epoch']}", flush=True)


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError("Existing manifest; inspect before restarting")
    old = read(ROOT.parent / "gan_fashion_width_01/manifest.json")
    reassessed = read(ROOT.parent / "fashion_classifier_upgrade_01/metrics.json")
    bundle = Path("data/gan_evaluation/fashion_mnist_v2")
    frozen = {**old["frozen_arguments"], "evaluation_set": str(bundle), "generator_lr": .0001, "discriminator_lr": .0002}
    evaluator = GeneratorEvaluator(PROJECT / bundle, torch.device("cpu"), latent_dim=100, conditional=False,
                                   seed=44, batch_size=128, confidence_threshold=.8, coverage_min_count=5, block_size=512)
    baselines = []
    for gc in (256, 512):
        prior = next(trial for trial in old["trials"] if trial["gc"] == gc and trial["dc"] == 64)
        metric = next(row for row in reassessed if row["id"] == prior["id"])
        baselines.append({**metric, "run": prior["run"], "completed_epochs": 30,
                          "selection": "v1-selected checkpoint, subsequently rescored with v2"})
    paths = [*sorted((PROJECT / "src/gan").glob("*.py")), *sorted((PROJECT / "src/shared").glob("*.py")),
             *(PROJECT / name for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock")),
             *(PROJECT / bundle / name for name in ("config.json", "classifier.pt", "reference.pt")),
             ROOT.parent / "gan_fashion_width_01/manifest.json", ROOT.parent / "fashion_classifier_upgrade_01/manifest.json",
             ROOT.parent / "fashion_classifier_upgrade_01/metrics.csv",
             *(PROJECT / trial["run"] / "best_kid.pt" for trial in old["trials"])]
    plan = {"status": "running", "scope": "Exactly two Fashion-MNIST runs; 30 epochs each; G .0001/D .0002",
            "started_utc": datetime.now(UTC).isoformat(), "frozen_arguments": frozen,
            "evaluation_protocol": evaluator.protocol, "baselines_v2": baselines,
            "comparison_limit": "Old checkpoints were selected with v1; a complete v2 epoch-selection control is unavailable",
            "input_interpretation": "User's 512/54 interpreted as previously discussed 512/64",
            "preflight": {"cuda_tests_passed": 6},
            "preserved_hashes": {path.relative_to(PROJECT).as_posix(): digest(path) for path in paths}, "trials": []}
    write(plan)
    for gc in (256, 512):
        run_trial(plan, gc)
    plan.update(status="complete", finished_utc=datetime.now(UTC).isoformat(),
                best_trial=min(plan["trials"], key=lambda trial: trial["kid"])["id"])
    write(plan)
    from export_analysis import main as export_analysis
    export_analysis()
    print("DONE: both full 30-epoch runs completed. No further training will start.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if MANIFEST.exists():
            plan = read(MANIFEST)
            plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
            write(plan)
        raise
