"""Execute the user-authorized MNIST sweep sequentially, with frozen KID controls.

Model/objective source files are never changed. Existing completed trials are
verified and reused, original checkpoints are preserved, and adaptations are
made only from completed validation-KID comparisons recorded in the manifest.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]
MANIFEST = ROOT / "manifest.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(plan: dict) -> None:
    temporary = ROOT / "manifest.tmp"
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def verify_frozen(plan: dict) -> None:
    for name, expected in plan["preserved_hashes"].items():
        if digest(PROJECT_ROOT / name) != expected:
            raise RuntimeError(f"Frozen source or evaluation artifact changed: {name}")
    reference = PROJECT_ROOT / plan["reference"]["run"] / "best_kid.pt"
    if digest(reference) != plan["reference"]["checkpoint_sha256"]:
        raise RuntimeError("Original reference checkpoint changed")


def config_key(settings: dict, epochs: int) -> tuple:
    return (settings["gc"], settings["dc"], settings["batch"], settings["g_lr"], settings["d_lr"], epochs)


def verify_trial(plan: dict, trial: dict) -> None:
    import math
    import torch
    directory = PROJECT_ROOT / trial["run"]
    config, history, summary = (read(directory / name) for name in ("config.json", "history.json", "summary.json"))
    args, settings = config["arguments"], trial["settings"]
    assert len(history) == args["epochs"] == trial["epochs"]
    assert config["generator"]["channels"] == settings["gc"] and config["discriminator"]["channels"] == settings["dc"]
    assert args["batch_size"] == settings["batch"] and args["generator_lr"] == settings["g_lr"] and args["discriminator_lr"] == settings["d_lr"]
    assert all(args[key] == value for key, value in plan["frozen_arguments"].items())
    assert config["evaluation_protocol"] == plan["evaluation_protocol"]
    assert config["split_sizes"] == {"train": 54000, "validation": 6000, "test": 10000}
    best = min(history, key=lambda row: row["evaluation"]["kid"])
    assert summary["best_epoch"] == best["epoch"] and summary["evaluation"] == best["evaluation"]
    assert all(math.isfinite(row["evaluation"]["kid"]) for row in history)
    assert all(row["evaluation"]["samples"] == row["evaluation"]["real_samples"] == 5000 for row in history)
    for row in history:
        for split in ("train", "validation"):
            assert all(math.isfinite(value) for value in row[split].values())
    inputs = torch.load(directory / "evaluation_inputs.pt", map_location="cpu", weights_only=True)
    reference = torch.load(PROJECT_ROOT / plan["reference"]["run"] / "evaluation_inputs.pt", map_location="cpu", weights_only=True)
    assert torch.equal(inputs["noise"], reference["noise"]) and inputs["labels"] is None
    checkpoint = torch.load(directory / "best_kid.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == summary["best_epoch"] and checkpoint["selection"]["score"] == best["evaluation"]["kid"]
    trial.update(best_epoch=summary["best_epoch"], kid=best["evaluation"]["kid"], metrics=best["evaluation"],
                 final_kid=history[-1]["evaluation"]["kid"], checkpoint_sha256=digest(directory / "best_kid.pt"),
                 logged_seconds=sum(row["seconds"] for row in history), summary=(directory / "summary.json").relative_to(PROJECT_ROOT).as_posix())


def run_trial(plan: dict, role: str, settings: dict, epochs: int, *, restart_suffix: str = "") -> dict:
    for trial in plan["trials"]:
        if config_key(trial["settings"], trial["epochs"]) == config_key(settings, epochs):
            if restart_suffix and trial["status"] == "interrupted_by_user":
                continue
            if trial["status"] != "complete":
                raise RuntimeError(f"Matching unfinished/failed trial requires inspection: {trial['id']}")
            verify_trial(plan, trial)
            if role not in trial["roles"]:
                trial["roles"].append(role)
            write(plan)
            print(f"REUSE {trial['id']}: KID {trial['kid']:.6f}", flush=True)
            return trial
    verify_frozen(plan)
    name = (f"gan_mnist_sweep01_gc{settings['gc']}_dc{settings['dc']}_b{settings['batch']}_"
            f"glr{round(settings['g_lr'] * 1e6)}u_dlr{round(settings['d_lr'] * 1e6)}u_kid_e{epochs}{restart_suffix}")
    output = Path("checkpoints/gan") / name
    if (PROJECT_ROOT / output).exists():
        raise FileExistsError(output)
    arguments = []
    for key, value in plan["frozen_arguments"].items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(option)
        elif value is not None:
            arguments.extend([option, str(value)])
    arguments.extend(["--epochs", str(epochs), "--batch-size", str(settings["batch"]),
                      "--generator-channels", str(settings["gc"]), "--discriminator-channels", str(settings["dc"]),
                      "--generator-lr", str(settings["g_lr"]), "--discriminator-lr", str(settings["d_lr"]),
                      "--output", output.as_posix()])
    trial = {"id": role, "roles": [role], "settings": dict(settings), "epochs": epochs,
             "run": output.as_posix(), "status": "running", "started_utc": datetime.now(UTC).isoformat(),
             "command": ["uv", "run", "--locked", "python", "-m", "gan.train", *arguments]}
    plan["trials"].append(trial)
    write(plan)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    print(f"START {role}: GC{settings['gc']}/DC{settings['dc']}, batch {settings['batch']}, "
          f"G/D lr {settings['g_lr']}/{settings['d_lr']}, {epochs} epochs", flush=True)
    started, seen_epoch = time.perf_counter(), 0
    with (logs / f"{role}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "gan.train", *arguments], cwd=PROJECT_ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        trial["pid"] = process.pid
        write(plan)
        while True:
            try:
                code = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                path = PROJECT_ROOT / output / "history.json"
                try:
                    history = read(path)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if len(history) != seen_epoch:
                    seen_epoch = len(history)
                    best = min(history, key=lambda row: row["evaluation"]["kid"])
                    trial["progress"] = {"epoch": seen_epoch, "kid": history[-1]["evaluation"]["kid"],
                                         "best_kid": best["evaluation"]["kid"], "best_epoch": best["epoch"]}
                    write(plan)
                    print(f"PROGRESS {role} {seen_epoch}/{epochs}: KID {history[-1]['evaluation']['kid']:.4f}, "
                          f"best {best['evaluation']['kid']:.4f} at {best['epoch']}", flush=True)
    trial.update(returncode=code, wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat())
    if code:
        trial["status"] = "failed"
        trial["log"] = (Path("logs") / f"{role}.log").as_posix()
        write(plan)
        print(f"FAILED {role}: exit {code}; log {trial['log']}", flush=True)
        return trial
    verify_trial(plan, trial)
    verify_frozen(plan)
    trial["status"] = "complete"
    write(plan)
    print(f"COMPLETE {role}: best KID {trial['kid']:.6f} at epoch {trial['best_epoch']}", flush=True)
    return trial


def choose(plan: dict, stage: str, candidates: list[dict]) -> dict:
    completed = [trial for trial in candidates if trial["status"] == "complete"]
    if not completed:
        raise RuntimeError(f"No completed candidates at {stage}")
    winner = min(completed, key=lambda trial: trial["kid"])
    plan["decisions"].append({"stage": stage, "candidates": [{"id": trial["id"], "kid": trial["kid"]} for trial in completed],
                              "winner": winner["id"], "settings": winner["settings"], "kid": winner["kid"]})
    write(plan)
    print(f"DECISION {stage}: {winner['id']}, KID {winner['kid']:.6f}", flush=True)
    return winner


def main() -> None:
    plan = read(MANIFEST)
    verify_frozen(plan)
    plan["status"] = "running"
    write(plan)
    base = plan["base"]
    control = {"id": "existing_gc128_dc64_b64_e60", "settings": base, "epochs": 60, "status": "complete",
               "kid": plan["reference"]["summary"]["evaluation"]["kid"], "run": plan["reference"]["run"],
               "metrics": plan["reference"]["summary"]["evaluation"]}
    initial = run_trial(plan, "initial_100", base, 100)
    candidates = [control]
    for batch in (128, 256):
        candidates.append(run_trial(plan, f"batch_{batch}_60", {**base, "batch": batch}, 60))
    for dc in (64, 128):
        candidates.append(run_trial(plan, f"width_gc256_dc{dc}_60", {**base, "gc": 256, "dc": dc}, 60))
    architecture = choose(plan, "architecture_and_batch_60", candidates)
    confirmation = run_trial(plan, "architecture_winner_100", architecture["settings"], 100)
    lr_base = choose(plan, "learning_rate_base_100", [initial, confirmation])
    matched_control = next(trial for trial in candidates if config_key(trial["settings"], 60) == config_key(lr_base["settings"], 60))
    rate_candidates = [matched_control]
    rates = [(0.0002, .0001), (.0002, .0004), (.0001, .0001), (.0001, .0002),
             (.0001, .0004), (.0004, .0001), (.0004, .0002), (.0004, .0004)]
    for g_lr, d_lr in rates:
        settings = {**lr_base["settings"], "g_lr": g_lr, "d_lr": d_lr}
        role = f"lr_g{round(g_lr * 1e6)}u_d{round(d_lr * 1e6)}u_60"
        rate_candidates.append(run_trial(plan, role, settings, 60))
    rates_winner = choose(plan, "learning_rates_60", rate_candidates)
    if config_key(rates_winner["settings"], 60) != config_key(lr_base["settings"], 60):
        run_trial(plan, "learning_rate_winner_100", rates_winner["settings"], 100)
    else:
        plan["decisions"].append({"stage": "learning_rate_winner_100", "action": "not needed; baseline rates retained"})
    verify_frozen(plan)
    plan.update(status="complete", finished_utc=datetime.now(UTC).isoformat())
    write(plan)
    print("SWEEP COMPLETE: all requested comparisons and conditional 100-epoch confirmations finished.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        plan = read(MANIFEST)
        plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        write(plan)
        raise
