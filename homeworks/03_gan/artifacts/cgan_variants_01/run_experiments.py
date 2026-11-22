"""Six user-authorized 30-epoch cGAN ablations; preserve earlier experiments."""

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
VARIANTS = (("patch", "patch", 0.), ("fm10", "global", 10.), ("patch_fm10", "patch", 10.))


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


def options(baseline: dict, variant: tuple, output: Path, smoke: bool) -> dict:
    args = dict(read(PROJECT / baseline["run"] / "config.json")["arguments"])
    args.update(epochs=1 if smoke else 30, output=output.as_posix(), resume=None,
                discriminator=variant[1], feature_matching_weight=variant[2],
                train_limit=128 if smoke else None, validation_limit=64 if smoke else None,
                download=False, device="cuda")
    return args


def command(args: dict) -> list[str]:
    result = ["-m", "gan.train"]
    for key, value in args.items():
        if value is None or value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.append(str(value))
    return result


def verify(plan: dict, trial: dict) -> dict:
    directory = PROJECT / trial["run"]
    baseline = next(item for item in plan["baselines"] if item["dataset"] == trial["dataset"])
    source = PROJECT / baseline["run"]
    before = read(source / "config.json")
    config, history, summary = (read(directory / name) for name in ("config.json", "history.json", "summary.json"))
    smoke = trial["kind"] == "smoke"
    epochs, steps_per_epoch = (1, 2) if smoke else (30, 844)
    assert [row["epoch"] for row in history] == list(range(1, epochs + 1))
    assert summary["last_epoch"] == epochs and summary["start_epoch"] == 0
    assert summary["g_updates"] == summary["d_updates"] == epochs * steps_per_epoch
    expected_args = {**trial["arguments"], "output": str(Path(trial["arguments"]["output"]))}
    assert config["arguments"] == expected_args
    for key in ("generator", "evaluation_protocol", "data_split_seed", "class_names", "preprocessing"):
        assert config[key] == before[key]
    if not smoke:
        assert config["split_sizes"] == before["split_sizes"] == {"train": 54000, "validation": 6000, "test": 10000}
    for name in ("preview_inputs.pt", "evaluation_inputs.pt"):
        original = torch.load(source / name, map_location="cpu", weights_only=True)
        current = torch.load(directory / name, map_location="cpu", weights_only=True)
        assert torch.equal(original["noise"], current["noise"])
        assert torch.equal(original["labels"], current["labels"])
    preview = torch.load(directory / "preview_inputs.pt", map_location="cpu", weights_only=True)
    assert preview["noise"].shape == (100, 100)
    assert torch.equal(preview["labels"], torch.arange(10).repeat_interleave(10))
    for row in history:
        evaluation = row["evaluation"]
        assert evaluation["samples"] == evaluation["real_samples"] == 5000
        assert math.isfinite(evaluation["kid"]) and 0 <= evaluation["conditional_accuracy"] <= 1
        assert row["train"]["images"] == (128 if smoke else 54000)
        assert row["validation"]["images"] == (64 if smoke else 6000)
        assert all(math.isfinite(value) for split in ("train", "validation") for value in row[split].values())
    selected = min(history, key=lambda row: row["evaluation"]["kid"])
    assert summary["best_epoch"] == selected["epoch"] and summary["evaluation"] == selected["evaluation"]
    checkpoint = torch.load(directory / "best_kid.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == selected["epoch"] and checkpoint["selection"]["score"] == selected["evaluation"]["kid"]
    assert checkpoint["training_state"]["global_g_step"] == selected["epoch"] * steps_per_epoch
    for key in ("generator_optimizer", "discriminator_optimizer"):
        assert all(state["step"].item() == selected["epoch"] * steps_per_epoch for state in checkpoint[key]["state"].values())
    expected_previews = {1, epochs, *range(5, epochs + 1, 5)}
    assert set(summary["sampled_epochs"]) == expected_previews
    assert all((directory / f"samples_epoch_{epoch:03d}.png").is_file() for epoch in expected_previews)
    return {"best_epoch": selected["epoch"], "metrics": selected["evaluation"],
            "last_metrics": history[-1]["evaluation"], "checkpoint_sha256": digest(directory / "best_kid.pt")}


def run(plan: dict, baseline: dict, variant: tuple, *, smoke: bool) -> None:
    protect(plan)
    dataset, name = baseline["dataset"], variant[0]
    output = (Path("runs/cgan_variants_01") / f"smoke_{dataset}_{name}" if smoke else
              Path("checkpoints/gan") / f"cgan_{dataset}_{'v2_' if dataset == 'fashion_mnist' else ''}gc256_dc64_b64_glr100u_dlr200u_{name}_kid_e30")
    if (PROJECT / output).exists():
        raise FileExistsError(output)
    args = options(baseline, variant, output, smoke)
    cmd = command(args)
    trial = {"dataset": dataset, "variant": name, "kind": "smoke" if smoke else "full",
             "run": output.as_posix(), "arguments": args,
             "command": ["uv", "run", "--locked", "python", *cmd],
             "status": "running", "started_utc": datetime.now(UTC).isoformat()}
    plan["trials"].append(trial)
    write(plan)
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"START {trial['kind']} {dataset} {name}: {args['epochs']} epochs", flush=True)
    begun, seen = time.perf_counter(), 0
    with (ROOT / "logs" / f"{trial['kind']}_{dataset}_{name}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, *cmd], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
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
                        print(f"PROGRESS {dataset} {name} {last['epoch']}/{args['epochs']}: "
                              f"KID {last['evaluation']['kid']:.4f}, labels {last['evaluation']['conditional_accuracy']:.1%}; "
                              f"best {best['evaluation']['kid']:.4f} at {best['epoch']}", flush=True)
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    trial.update(returncode=code, wall_seconds=time.perf_counter() - begun, finished_utc=datetime.now(UTC).isoformat())
    if code:
        trial["status"] = "failed"
        write(plan)
        raise RuntimeError(f"Training failed; inspect {trial['kind']}_{dataset}_{name}.log")
    trial.update(verify(plan, trial), status="complete")
    if not smoke:
        destination = ROOT / f"{dataset}_{name}_samples_10x10.png"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(PROJECT / output / "samples_best_kid.png", destination)
        trial["grid"] = destination.relative_to(PROJECT).as_posix()
        trial["grid_sha256"] = digest(destination)
    protect(plan)
    write(plan)
    print(f"COMPLETE {trial['kind']} {dataset} {name}: best KID {trial['metrics']['kid']:.4f}, "
          f"epoch {trial['best_epoch']}, labels {trial['metrics']['conditional_accuracy']:.1%}", flush=True)


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError("Existing manifest: inspect before restarting any trials")
    original = read(PROJECT / "homeworks/03_gan/artifacts/cgan_baseline_01/manifest.json")
    assert original["status"] == "complete"
    baselines = []
    protected = [PROJECT / "README.md", PROJECT / "AGENTS.md", PROJECT / "pyproject.toml", PROJECT / "uv.lock",
                 *sorted((PROJECT / "src/gan").glob("*.py")), *sorted((PROJECT / "src/shared").glob("*.py"))]
    for trial in original["trials"]:
        if trial["kind"] != "full":
            continue
        directory = PROJECT / trial["run"]
        history = [row for row in read(directory / "history.json") if row["epoch"] <= 30]
        assert len(history) == 30
        selected = min(history, key=lambda row: row["evaluation"]["kid"])
        grid = directory / f"samples_epoch_{selected['epoch']:03d}.png"
        if not grid.is_file():
            raise FileNotFoundError("Historical best-30 baseline grid is unavailable; do not substitute the best-60 grid")
        destination = ROOT / f"{trial['dataset']}_baseline_samples_10x10.png"
        shutil.copyfile(grid, destination)
        baselines.append({"dataset": trial["dataset"], "run": trial["run"], "best_epoch": selected["epoch"],
                          "metrics": selected["evaluation"], "selection_budget": 30,
                          "grid": destination.relative_to(PROJECT).as_posix(),
                          "grid_source": grid.relative_to(PROJECT).as_posix(), "grid_sha256": digest(destination),
                          "checkpoint_available": selected["epoch"] == trial["best_epoch"]})
        protected.extend(path for path in directory.rglob("*") if path.is_file())
        config = read(directory / "config.json")
        protected.extend(path for path in (PROJECT / config["arguments"]["evaluation_set"]).rglob("*") if path.is_file())
    plan = {"status": "running", "started_utc": datetime.now(UTC).isoformat(), "trials": [], "baselines": baselines,
            "checks": {"model_training_evaluation_metrics": "54 passed in 21.37s"},
            "selection_budget": 30, "feature_matching_weight": 10.,
            "protected_hashes": {path.relative_to(PROJECT).as_posix(): digest(path) for path in protected},
            "notes": ["Exactly six full runs: PatchGAN, feature matching weight 10, and both; each dataset.",
                      "All fresh initializations; identical cGAN settings, fixed noise/labels and frozen validation bundles.",
                      "Baseline metrics use only first 30 epochs, and grids are historical best-30 previews.",
                      "Only best-60 baseline weights were retained; do not label those as best-30 weights.",
                      "Feature matching is additive batch-mean discriminator-feature MSE, not classwise matching.",
                      "KID scores are comparable within dataset only; classifier agreement is a diagnostic.",
                      "One seed with nondeterministic CUDA; no test-set use or 60-epoch extensions."]}
    write(plan)
    try:
        for baseline in baselines:
            for variant in VARIANTS:
                run(plan, baseline, variant, smoke=True)
        for baseline in baselines:
            for variant in VARIANTS:
                run(plan, baseline, variant, smoke=False)
        protect(plan)
        plan.update(status="training_complete", finished_utc=datetime.now(UTC).isoformat())
        write(plan)
        commands = [" ".join(trial["command"]) for trial in plan["trials"] if trial["kind"] == "full"]
        (ROOT / "commands.txt").write_text("\n\n".join(commands) + "\n", encoding="utf-8")
        from export_analysis import main as export_analysis
        export_analysis()
        plan.update(status="complete", analysis_finished_utc=datetime.now(UTC).isoformat())
        write(plan)
        print("DONE: six 30-epoch experiments completed and comparison artifacts exported. No further training.", flush=True)
    except BaseException as error:
        plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        write(plan)
        raise


if __name__ == "__main__":
    main()
