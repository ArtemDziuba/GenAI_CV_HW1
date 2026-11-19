"""Evaluate a selected checkpoint and export generation experiments without training."""

import argparse
from pathlib import Path
from uuid import uuid4

import matplotlib
import torch

from shared import make_dataloaders, seed_everything
from shared.experiment import limit_loader, resolve_device, write_json

from .artifacts import save_reconstructions, save_samples, save_traversal
from .models import VAE
from .train import load_checkpoint
from .training import importance_evaluation, run_epoch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh artifact directory")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Smoke evaluation only; omit for final metrics")
    parser.add_argument("--importance-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--reconstruction-count", type=int, default=24)
    parser.add_argument("--reconstruction-columns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0, help="Display samples only; never changes metrics")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.importance_samples, args.sample_count,
           args.reconstruction_count, args.reconstruction_columns) < 1 or not 0 < args.temperature < float("inf"):
        parser.error("counts must be positive; temperature finite and positive")
    matplotlib.use("Agg")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    training_args = config["arguments"]
    loaders = make_dataloaders(
        training_args["dataset"], args.data_root, batch_size=args.batch_size,
        validation_fraction=training_args["validation_fraction"], seed=training_args["seed"],
        pin_memory=device.type == "cuda", download=False,
    )
    # Preserve smoke-run validation membership when reproducing its metric.
    count = training_args.get("validation_limit") if args.split == "validation" else None
    loader = limit_loader(getattr(loaders, args.split), count, shuffle=False)
    loader = limit_loader(loader, args.limit, shuffle=False)
    output = args.output or args.checkpoint.parent / f"evaluation_{args.split}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    options = (config["objective"] if isinstance(model, VAE) else
               {"ae_loss": config["objective"].get("ae_loss", "mse")})
    metrics = run_epoch(model, loader, device, seed=args.seed, progress=True, **options)
    if isinstance(model, VAE):
        metrics.update(importance_evaluation(
            model, loader, device, samples=args.importance_samples, seed=args.seed, progress=True,
            likelihood=options["likelihood"], observation_std=options["observation_std"],
        ))
        save_samples(model, output / "samples.png", seed=args.seed, count=args.sample_count, temperature=args.temperature)
        save_samples(model, output / "observations.png", seed=args.seed, count=args.sample_count,
                     temperature=args.temperature, likelihood=options["likelihood"],
                     observation_std=options["observation_std"], sample_observations=True)
        save_traversal(model, output / "traversal.png")
    preview = torch.stack([loader.dataset[index][0]
                           for index in range(min(args.reconstruction_count, len(loader.dataset)))])
    save_reconstructions(model, preview, output / "reconstructions.png",
                         likelihood=options.get("likelihood", "gaussian"), seed=args.seed,
                         count=args.reconstruction_count, nrow=args.reconstruction_columns)
    write_json(output / "evaluation.json", {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split, "count": len(loader.dataset), "metrics": metrics,
        "seed": args.seed, "sample_temperature": args.temperature,
        "reconstruction_count": len(preview), "reconstruction_columns": args.reconstruction_columns,
        "observation_protocol": config["observation_protocol"], "units": config["units"],
        "likelihood_note": "negative ELBO/IWAE bounds are estimates, not exact marginal NLL",
    })
    print(metrics)
    print(f"Saved evaluation and figures: {output}")


if __name__ == "__main__":
    main()
