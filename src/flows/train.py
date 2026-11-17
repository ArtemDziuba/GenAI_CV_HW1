"""Run an explicitly requested MNIST/Fashion-MNIST baseline: python -m flows.train."""

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from shared import get_device, make_dataloaders, plot_image_grid, plot_metrics, seed_everything

from .likelihood import RealNVPDensity
from .realnvp import RealNVP
from .training import run_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="New run directory; existing paths are refused")
    parser.add_argument("--download", action="store_true", help="Explicitly allow dataset downloads")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--conditioner-depth", type=int, default=3,
                        help="Convolutions inside each coupling CNN, including input/output (minimum 3)")
    parser.add_argument("--num-scales", type=int, default=2)
    parser.add_argument("--couplings-per-scale", type=int, default=3,
                        help="Couplings per mask family, per scale (default: 12 layers total)")
    parser.add_argument("--scale-limit", type=float, default=2.0)
    parser.add_argument("--logit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit-alpha", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--train-limit", type=int, help="Use only this many training examples for a smoke run")
    parser.add_argument("--validation-limit", type=int, help="Limit validation examples for a smoke run")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the best validation checkpoint once on the official test set")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    return get_device() if name == "auto" else torch.device(name)


def limit_loader(loader: DataLoader, count: int | None, *, shuffle: bool) -> DataLoader:
    """Keep a prefix of an already seeded training/validation split, if requested."""
    if count is None:
        return loader
    if not 1 <= count <= len(loader.dataset):
        raise ValueError("subset size must be positive and no larger than its split")
    return DataLoader(
        Subset(loader.dataset, range(count)), batch_size=loader.batch_size,
        shuffle=shuffle, num_workers=loader.num_workers, pin_memory=loader.pin_memory,
        worker_init_fn=loader.worker_init_fn, generator=loader.generator, drop_last=False,
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_best_checkpoint(
    path: Path, model: RealNVPDensity, optimizer: torch.optim.Optimizer, *,
    epoch: int, validation: dict[str, float], config: dict,
) -> None:
    """Replace only this new run's selected checkpoint, with an atomic rename."""
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1,
        "epoch": epoch,
        "validation": validation,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, temporary)
    temporary.replace(path)


def load_density_checkpoint(path: Path, device: torch.device) -> tuple[RealNVPDensity, dict]:
    """Reconstruct a baseline from saved config; no pickled model objects needed."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    flow = RealNVP(**config["model"])
    model = RealNVPDensity(flow, **config["preprocessing"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def save_samples(model: RealNVPDensity, output: Path, *, seed: int, count: int) -> None:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = model.sample(count, generator=generator)
    figure = plot_image_grid(samples, max_images=count, save_path=output)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    matplotlib.use("Agg")  # CLI artifact saving; importing helpers keeps notebook backends intact.
    for name in ("epochs", "batch_size", "sample_count", "sample_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.conditioner_depth < 3:
        parser.error("--conditioner-depth must be at least 3")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive")
    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=args.deterministic)
    flow_config = {
        "input_shape": (1, 28, 28), "hidden_channels": args.hidden_channels,
        "num_scales": args.num_scales, "couplings_per_scale": args.couplings_per_scale,
        "scale_limit": args.scale_limit,
        "conditioner_depth": args.conditioner_depth,
    }
    preprocess_config = {"logit": args.logit, "alpha": args.logit_alpha, "bits": 8}
    model = RealNVPDensity(RealNVP(**flow_config), **preprocess_config).to(device)
    output = args.output or Path("checkpoints") / f"realnvp_{args.dataset}_{uuid4().hex[:8]}"
    # Each invocation must use a fresh directory to protect existing experiments.
    output.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}; PyTorch: {torch.__version__}; output: {output}", flush=True)
    loaders = make_dataloaders(
        args.dataset, args.data_root, batch_size=args.batch_size,
        validation_fraction=args.validation_fraction, seed=args.seed,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", download=args.download,
    )
    train_loader = limit_loader(loaders.train, args.train_limit, shuffle=True)
    validation_loader = limit_loader(loaders.validation, args.validation_limit, shuffle=False)
    arguments = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()}
    config = {
        "arguments": arguments, "model": flow_config, "preprocessing": preprocess_config,
        "runtime": {"device": str(device), "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "split_sizes": {"train": len(train_loader.dataset), "validation": len(validation_loader.dataset),
                        "test": len(loaders.test.dataset)},
        "likelihood": "uniform-dequantization bound; nats/image; 8-bit bin correction included",
    }
    write_json(output / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict] = []
    best_nll, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, train_loader, device, optimizer=optimizer,
                          seed=args.seed + epoch, progress=True)
        validation = run_epoch(model, validation_loader, device, seed=args.seed, progress=True)
        row = {"epoch": epoch, "train_nll": train["nll"], "train_bpd": train["bpd"],
               "validation_nll": validation["nll"], "validation_bpd": validation["bpd"],
               "seconds": time.perf_counter() - start}
        history.append(row)
        if validation["nll"] < best_nll:
            best_nll, best_epoch = validation["nll"], epoch
            save_best_checkpoint(output / "best.pt", model, optimizer, epoch=epoch,
                                 validation=validation, config=config)
        write_json(output / "history.json", history)
        figure = plot_metrics(
            {"train": [row["train_bpd"] for row in history],
             "validation": [row["validation_bpd"] for row in history]},
            ylabel="Bits per dimension (dequantization bound)", save_path=output / "losses.png",
        )
        plt.close(figure)
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(model, output / f"samples_epoch_{epoch:03d}.png",
                         seed=args.seed, count=args.sample_count)
        print(f"Epoch {epoch}/{args.epochs}: train {train['bpd']:.4f} BPD; "
              f"validation {validation['bpd']:.4f} BPD; best epoch {best_epoch}", flush=True)
    best_model, checkpoint = load_density_checkpoint(output / "best.pt", device)
    save_samples(best_model, output / "samples_best.png", seed=args.seed, count=args.sample_count)
    summary = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"]}
    if args.evaluate_test:
        summary["test"] = run_epoch(best_model, loaders.test, device, seed=args.seed + 1, progress=True)
    write_json(output / "summary.json", summary)
    print(f"Finished. Best checkpoint and artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
