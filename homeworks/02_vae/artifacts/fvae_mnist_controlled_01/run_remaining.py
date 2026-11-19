"""Run only the four remaining experiments authorized in this session."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def write_manifest(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=int, choices=(1, 4), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifacts = Path(__file__).resolve().parent
    root = next(path for path in artifacts.parents if (path / "pyproject.toml").exists())
    manifest_path = artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = manifest["reference_config"]
    plans = [("fh128", 4, 128), ("fh32", 4, 32), ("f8", 8, 64), ("f2", 2, 64)]
    commands = []
    for label, steps, hidden in plans:
        name = f"fvae_mnist_z32_c32_h256_d1_f{steps}_fh{hidden}_beta{args.beta}_e30"
        output = Path("checkpoints/vae") / name
        settings = dict(reference["arguments"])
        settings.update(output=output.as_posix(), beta=args.beta,
                        flow_steps=steps, flow_hidden_dim=hidden)
        command = ["uv", "--cache-dir", ".uv-cache", "run", "--locked", "--offline",
                   "python", "-m", "vae.train"]
        for key, value in settings.items():
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            command.append(flag)
            if value is not True:
                command.append(str(value))
        if (root / output).exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        commands.append((label, output, command))
    if args.dry_run:
        from vae.train import build_parser
        for label, output, command in commands:
            parsed = vars(build_parser().parse_args(command[9:]))
            expected = dict(reference["arguments"])
            for key, value in parsed.items():
                if key in ("output", "flow_steps", "flow_hidden_dim", "beta"):
                    continue
                original = expected[key]
                if isinstance(value, Path):
                    assert value == Path(original), (key, value, original)
                else:
                    assert value == original, (key, value, original)
            print(label, subprocess.list2cmdline(command))
        return
    if not manifest["beta_selection"] or manifest["beta_selection"]["chosen_beta"] != args.beta:
        raise RuntimeError("Record the completed beta comparison before running the remaining experiments")
    environment = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    for label, output, command in commands:
        entry = {"label": label, "output": output.as_posix(), "command": command,
                 "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
        manifest["runs"].append(entry)
        manifest["status"] = "running_" + label
        write_manifest(manifest_path, manifest)
        print(f"Starting {label}: {output}", flush=True)
        try:
            with (artifacts / f"{label}.log").open("wb") as handle:
                subprocess.run(command, cwd=root, env=environment, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            config = json.loads((root / output / "config.json").read_text(encoding="utf-8"))
            for key, value in reference["model"].items():
                if key not in ("flow_steps", "flow_hidden_dim"):
                    assert config["model"][key] == value, key
            assert config["objective"] == dict(reference["objective"], beta=float(args.beta))
            entry.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                         summary=json.loads((root / output / "summary.json").read_text(encoding="utf-8")))
            write_manifest(manifest_path, manifest)
            print(f"Completed {label}: {entry['summary']}", flush=True)
        except Exception as error:
            entry.update(status="failed", error=str(error))
            manifest["status"] = "failed_" + label
            write_manifest(manifest_path, manifest)
            raise
    manifest["status"] = "training_complete"
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
