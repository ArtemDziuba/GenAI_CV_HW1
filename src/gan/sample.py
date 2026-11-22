"""Generate comparable samples from any saved GAN checkpoint, without training."""

import argparse
from pathlib import Path
from uuid import uuid4

import matplotlib
import torch

from shared.experiment import resolve_device, write_json
from .artifacts import fixed_inputs, save_samples
from .train import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh directory")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.sample_count, args.samples_per_class) < 1 or not 0 <= args.seed < 2**32:
        parser.error("sample counts must be positive; seed must be in [0,2**32)")
    output = args.output or args.checkpoint.parent / f"preview_{uuid4().hex[:8]}"
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    matplotlib.use("Agg")
    generator, _, checkpoint = load_checkpoint(args.checkpoint, resolve_device(args.device))
    noise, labels = fixed_inputs(generator, count=args.sample_count, samples_per_class=args.samples_per_class, seed=args.seed)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = save_samples(generator, noise, labels, output / "samples.png",
                               class_names=checkpoint["config"]["class_names"],
                               columns=args.samples_per_class if generator.conditional else 8)
    torch.save({"noise": noise, "labels": labels}, output / "preview_inputs.pt")
    write_json(output / "samples.json", {"checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
                                        "seed": args.seed, "count": len(noise), "requested_labels": labels.tolist() if labels is not None else None,
                                        "diagnostics": diagnostics, "note": "Pixel spread is not class coverage or proof against mode collapse."})
    print(f"Saved {len(noise)} samples: {output}")


if __name__ == "__main__":
    main()
