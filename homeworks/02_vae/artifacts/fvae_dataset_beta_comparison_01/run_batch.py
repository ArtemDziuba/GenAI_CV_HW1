"""Run the explicitly requested two-dataset f-VAE comparison, preserving old runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


FOLDER = Path(__file__).resolve().parent
ROOT = next(path for path in FOLDER.parents if (path / "pyproject.toml").exists())
MANIFEST = FOLDER / "manifest.json"


def write_manifest(value):
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def assert_settings(config, reference, dataset, beta, steps, width):
    expected = dict(reference["model"], flow_steps=steps, flow_hidden_dim=width)
    assert config["kind"] == "fvae" and config["model"] == expected
    assert config["objective"] == dict(reference["objective"], beta=float(beta))
    settings = dict(reference["arguments"], dataset=dataset, beta=beta,
                    flow_steps=steps, flow_hidden_dim=width)
    for key, value in settings.items():
        if key != "output":
            assert config["arguments"][key] == value, (key, config["arguments"][key], value)


def make_plan(manifest):
    reference = manifest["reference_config"]
    plan = []
    for dataset, betas in (("mnist", (1,)), ("fashion_mnist", (1, 4))):
        for beta in betas:
            for steps, width in ((4, 32), (4, 64), (2, 64)):
                name = f"fvae_{dataset}_z32_c32_h256_d1_f{steps}_fh{width}_beta{beta}_e30"
                canonical = Path("checkpoints/vae") / name
                output = canonical
                reused = False
                if (ROOT / output / "summary.json").exists() and (ROOT / output / "best.pt").exists():
                    config = json.loads((ROOT / output / "config.json").read_text(encoding="utf-8"))
                    assert_settings(config, reference, dataset, beta, steps, width)
                    history = json.loads((ROOT / output / "history.json").read_text(encoding="utf-8"))
                    assert len(history) == 30
                    reused = True
                else:
                    repeat = 2
                    while (ROOT / output).exists():
                        output = canonical.with_name(f"{name}_r{repeat}")
                        repeat += 1
                settings = dict(reference["arguments"], dataset=dataset, beta=beta,
                                flow_steps=steps, flow_hidden_dim=width, output=output.as_posix())
                command = ["uv", "--cache-dir", ".uv-cache", "run", "--locked", "--offline",
                           "python", "-m", "vae.train"]
                for key, value in settings.items():
                    if value is None or value is False:
                        continue
                    command.append("--" + key.replace("_", "-"))
                    if value is not True:
                        command.append(str(value))
                entry = {"label": f"{dataset}_beta{beta}_f{steps}_fh{width}",
                         "dataset": dataset, "beta": beta, "flow_steps": steps,
                         "flow_hidden_dim": width, "output": output.as_posix(), "command": command,
                         "status": "reused" if reused else "pending", "reused": reused}
                if reused:
                    entry["checkpoint_sha256"] = hashlib.sha256((ROOT / output / "best.pt").read_bytes()).hexdigest()
                    entry["summary"] = json.loads((ROOT / output / "summary.json").read_text(encoding="utf-8"))
                plan.append(entry)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["experiments"]:
        raise RuntimeError("This bounded batch has already been launched; inspect its manifest before repeating")
    plan = make_plan(manifest)
    from vae.train import build_parser
    for entry in plan:
        parsed = build_parser().parse_args(entry["command"][9:])
        assert parsed.model == "fvae" and parsed.epochs == 30
        print(f"{'REUSE' if entry['reused'] else 'TRAIN'} {entry['label']}: {entry['output']}", flush=True)
    assert len(plan) == 9
    if args.dry_run:
        return
    manifest["experiments"] = plan
    manifest["status"] = "running"
    manifest["started_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest)
    environment = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    for entry in manifest["experiments"]:
        if entry["reused"]:
            continue
        assert not (ROOT / entry["output"]).exists()
        entry.update(status="running", started_utc=datetime.now(timezone.utc).isoformat())
        manifest["active_run"] = entry["label"]
        write_manifest(manifest)
        print(f"Starting {entry['label']}", flush=True)
        try:
            with (FOLDER / f"{entry['label']}.log").open("wb") as log:
                subprocess.run(entry["command"], cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            config = json.loads((ROOT / entry["output"] / "config.json").read_text(encoding="utf-8"))
            assert_settings(config, manifest["reference_config"], entry["dataset"], entry["beta"],
                            entry["flow_steps"], entry["flow_hidden_dim"])
            entry.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                         summary=json.loads((ROOT / entry["output"] / "summary.json").read_text(encoding="utf-8")))
            write_manifest(manifest)
            print(f"Completed {entry['label']}: {entry['summary']}", flush=True)
        except Exception as error:
            entry.update(status="failed", error=str(error))
            manifest["status"] = "failed"
            write_manifest(manifest)
            raise
    manifest.update(status="training_complete", active_run=None,
                    training_finished_utc=datetime.now(timezone.utc).isoformat())
    write_manifest(manifest)


if __name__ == "__main__":
    main()
