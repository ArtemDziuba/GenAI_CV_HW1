"""Export fixed-seed prior means from selected checkpoints, without training."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

from shared.experiment import resolve_device
from vae.artifacts import save_samples
from vae.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = resolve_device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for run in args.run_dir:
        model, checkpoint = load_checkpoint(run / "best.pt", device)
        folder = args.output_dir / run.name
        folder.mkdir(exist_ok=False)
        files = []
        for seed in (42, 123, 2026):
            path = folder / f"seed_{seed}.png"
            save_samples(model, path, seed=seed, count=64, temperature=1.0)
            files.append(path.as_posix())
        records.append({
            "run": run.as_posix(), "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
            "seeds": [42, 123, 2026], "count_per_seed": 64, "temperature": 1.0,
            "display": "decoder means; no observation noise", "files": files,
        })
        print(run.name, "exported 192 prior means", flush=True)
        del model
    (args.output_dir / "metadata.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
