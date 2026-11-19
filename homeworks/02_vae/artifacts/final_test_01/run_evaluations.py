"""Evaluate only the six frozen selections; reuse completed results, never train."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]


def write_manifest(plan: dict) -> None:
    temporary = ROOT / "manifest.tmp"
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(ROOT / "manifest.json")


def verify_result(selection: dict, result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["split"] == "test" and result["count"] == 10000
    assert result["seed"] == 42 and result["checkpoint_epoch"] == selection["checkpoint_epoch"]
    assert all(math.isfinite(value) for value in result["metrics"].values())
    assert Path(result["checkpoint"]).as_posix() == selection["run"] + "/best.pt"
    for name, key in (("best.pt", "checkpoint_sha256"), ("config.json", "config_sha256")):
        assert hashlib.sha256((PROJECT_ROOT / selection["run"] / name).read_bytes()).hexdigest() == selection[key]
    if selection["kind"] != "ae":
        assert result["metrics"]["importance_samples"] == 32
    return result


def main() -> None:
    plan = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert plan["selection_frozen"] and len(plan["selections"]) == 6
    for selection in plan["selections"]:
        output = ROOT / selection["output"]
        result_path = output / "evaluation.json"
        if selection["status"] == "complete":
            verify_result(selection, result_path)
            print(f"Reusing completed {selection['dataset']} {selection['kind']}", flush=True)
            continue
        command = [sys.executable, "-m", "vae.evaluate", "--checkpoint", selection["run"] + "/best.pt",
                   "--output", str(output.relative_to(PROJECT_ROOT)), "--device", "cuda", "--split", "test",
                   "--batch-size", "64", "--seed", "42", "--importance-samples", "32",
                   "--reconstruction-count", "24", "--reconstruction-columns", "8"]
        selection.update(status="running", command=command)
        write_manifest(plan)
        print(f"Starting {selection['dataset']} {selection['kind']}: 10,000 test images, IWAE K=32", flush=True)
        started = time.perf_counter()
        log_path = ROOT / f"step1_{selection['dataset']}_{selection['kind']}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if result.returncode:
            selection.update(status="failed", returncode=result.returncode)
            write_manifest(plan)
            raise RuntimeError(f"Evaluation failed: inspect {log_path}")
        evaluation = verify_result(selection, result_path)
        selection.update(status="complete", seconds=time.perf_counter() - started,
                         test=evaluation["metrics"], evaluation_file=result_path.relative_to(PROJECT_ROOT).as_posix())
        write_manifest(plan)
        print(f"Complete {selection['dataset']} {selection['kind']}: {evaluation['metrics']}", flush=True)
    plan["status"] = "complete"
    write_manifest(plan)
    print("Both steps complete: six unique checkpoint evaluations, reused for three matched dataset pairs.", flush=True)


if __name__ == "__main__":
    main()
