"""Continue the two user-selected Fashion checkpoints to total epoch 60."""

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


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
MANIFEST = ROOT / "manifest.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(plan: dict) -> None:
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def verify_protected(plan: dict) -> None:
    for name, expected in plan["protected_hashes"].items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError(f"Protected file changed: {name}")


def verify_trial(plan: dict, trial: dict) -> None:
    directory = PROJECT / trial["run"]
    parent = next(item for item in plan["parents"] if item["gc"] == trial["gc"])
    source_config = read(PROJECT / parent["run"] / "config.json")
    config, history, summary = (read(directory / name) for name in ("config.json", "history.json", "summary.json"))
    start = parent["epoch"]
    assert [row["epoch"] for row in history] == list(range(start + 1, 61))
    assert summary["last_epoch"] == 60 and summary["start_epoch"] == start
    assert summary["epochs_trained_this_run"] == 60 - start
    assert summary["g_updates"] == summary["d_updates"] == 60 * 844
    assert summary["session_g_updates"] == summary["session_d_updates"] == (60 - start) * 844
    for key in ("generator", "discriminator", "evaluation_protocol", "split_sizes", "data_split_seed"):
        assert config[key] == source_config[key]
    for key, value in source_config["arguments"].items():
        if key not in ("epochs", "output", "resume"):
            assert config["arguments"][key] == value
    assert config["resumption"]["optimizer_states_restored"]
    assert config["resumption"]["checkpoint_sha256"] == plan["protected_hashes"][parent["checkpoint"]]
    baseline = read(directory / "resume_baseline.json")
    assert baseline["epoch"] == start and abs(baseline["evaluation"]["kid"] - parent["kid"]) < 1e-4
    best = min([baseline, *history], key=lambda row: row["evaluation"]["kid"])
    assert summary["best_epoch"] == best["epoch"] and summary["evaluation"] == best["evaluation"]
    for row in history:
        assert row["evaluation"]["samples"] == row["evaluation"]["real_samples"] == 5000
        assert math.isfinite(row["evaluation"]["kid"])
        for split in ("train", "validation"):
            assert all(math.isfinite(value) for value in row[split].values())
    checkpoint = torch.load(directory / "best_kid.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == best["epoch"] and checkpoint["selection"]["score"] == best["evaluation"]["kid"]
    assert checkpoint["training_state"]["global_g_step"] == best["epoch"] * 844
    for key in ("generator_optimizer", "discriminator_optimizer"):
        assert all(state["step"].item() == best["epoch"] * 844 for state in checkpoint[key]["state"].values())
    for name in ("evaluation_inputs.pt", "preview_inputs.pt"):
        before = torch.load(PROJECT / parent["run"] / name, map_location="cpu", weights_only=True)
        after = torch.load(directory / name, map_location="cpu", weights_only=True)
        assert torch.equal(before["noise"], after["noise"]) and after["labels"] is None
    expected_previews = {start, start + 1, 60, *range(((start // 5) + 1) * 5, 61, 5)}
    assert set(summary["sampled_epochs"]) == expected_previews
    assert all((directory / f"samples_epoch_{epoch:03d}.png").is_file() for epoch in expected_previews)
    trial.update(start_epoch=start, final_epoch=60, epochs_trained=60 - start, best_epoch=best["epoch"],
                 kid=best["evaluation"]["kid"], metrics=best["evaluation"], final_kid=history[-1]["evaluation"]["kid"],
                 checkpoint_sha256=digest(directory / "best_kid.pt"), session_g_updates=summary["session_g_updates"])


def run_trial(plan: dict, parent: dict) -> None:
    verify_protected(plan)
    start, gc = parent["epoch"], parent["gc"]
    output = Path("checkpoints/gan") / f"gan_fashion_mnist_v2_gc{gc}_dc64_glr100u_dlr200u_resume_e{start}_to60"
    if (PROJECT / output).exists():
        raise FileExistsError(output)
    args = ["--resume", parent["checkpoint"], "--evaluation-set", "data/gan_evaluation/fashion_mnist_v2",
            "--epochs", "60", "--device", "cuda", "--output", output.as_posix()]
    trial = {"id": f"gc{gc}_dc64", "gc": gc, "run": output.as_posix(), "status": "running", "source_epoch": start,
             "command": ["uv", "run", "--locked", "python", "-m", "gan.train", *args],
             "started_utc": datetime.now(UTC).isoformat()}
    plan["trials"].append(trial)
    write(plan)
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"START G{gc}/D64: restore best epoch {start}; continue {start + 1}..60 ({60 - start} new epochs)", flush=True)
    begun, seen = time.perf_counter(), 0
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
                    baseline = read(PROJECT / output / "resume_baseline.json")
                    best = min([baseline, *history], key=lambda row: row["evaluation"]["kid"])
                    trial["progress"] = {"epoch": history[-1]["epoch"], "new_epochs_completed": seen,
                                         "kid": history[-1]["evaluation"]["kid"], "best_epoch": best["epoch"], "best_kid": best["evaluation"]["kid"]}
                    write(plan)
                    print(f"PROGRESS G{gc}/D64 epoch {history[-1]['epoch']}/60: v2 KID {history[-1]['evaluation']['kid']:.4f}; "
                          f"best {best['evaluation']['kid']:.4f} at {best['epoch']}", flush=True)
    trial.update(returncode=code, wall_seconds=time.perf_counter() - begun, finished_utc=datetime.now(UTC).isoformat())
    if code:
        trial["status"] = "failed"
        write(plan)
        raise RuntimeError(f"Continuation failed for {trial['id']}; inspect its log")
    verify_trial(plan, trial)
    verify_protected(plan)
    trial["status"] = "complete"
    write(plan)
    print(f"COMPLETE G{gc}/D64: reached epoch 60; best KID {trial['kid']:.6f} at {trial['best_epoch']}", flush=True)


def main() -> None:
    plan = read(MANIFEST)
    if plan["status"] != "implementing_resume":
        raise RuntimeError("Existing continuation manifest needs inspection before restarting")
    verify_protected(plan)
    sources = [*sorted((PROJECT / "src/gan").glob("*.py")), *sorted((PROJECT / "src/shared").glob("*.py"))]
    plan["protected_hashes"].update({path.relative_to(PROJECT).as_posix(): digest(path) for path in sources})
    plan.update(status="running", started_utc=datetime.now(UTC).isoformat(), trials=[],
                note="Legacy RNG state unavailable; branches restore paired networks, optimizers and absolute epoch numbering")
    write(plan)
    for parent in plan["parents"]:
        run_trial(plan, parent)
    plan.update(status="complete", finished_utc=datetime.now(UTC).isoformat())
    write(plan)
    from export_analysis import main as export_analysis
    export_analysis()
    print("DONE: both requested continuations reached total epoch 60. No further training will start.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        plan = read(MANIFEST)
        plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        write(plan)
        raise
