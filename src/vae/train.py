"""Train an AE, VAE, or posterior-flow VAE: python -m vae.train --help."""

import argparse
from pathlib import Path
import time
from uuid import uuid4

import matplotlib
import torch

from shared import make_dataloaders, seed_everything
from shared.experiment import limit_loader, resolve_device, write_json

from .artifacts import save_history, save_reconstructions, save_samples, save_traversal
from .models import Autoencoder, VAE, build_model
from .objectives import validate_likelihood
from .training import run_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ae", "vae", "fvae"], default="vae")
    parser.add_argument("--ae-loss", choices=["mse", "l1"], default="mse",
                        help="AE reconstruction objective; checkpoint selection remains validation MSE")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Fresh run directory; existing paths are refused")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay; off by default")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32, help="Encoder widths C,2C; decoder mirrors them")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dense feature width before latent heads")
    parser.add_argument("--conv-depth", type=int, default=1, help="Convolutions per resolution stage")
    parser.add_argument("--flow-steps", type=int, default=4, help="Alternating latent couplings; fvae only")
    parser.add_argument("--flow-hidden-dim", type=int, default=128)
    parser.add_argument("--flow-hidden-layers", type=int, default=2)
    parser.add_argument("--scale-limit", type=float, default=1.0, help="Bound on each coupling log-scale")
    parser.add_argument("--likelihood", choices=["bernoulli", "gaussian"], default="bernoulli")
    parser.add_argument("--observation-std", type=float, default=0.1, help="Fixed Gaussian observation std; not latent std")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight; beta!=1 is optional beta-VAE")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient norm limit; 0 disables clipping")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--reconstruction-count", type=int, default=24)
    parser.add_argument("--reconstruction-columns", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--evaluate-test", action="store_true", help="Only use for final selected configurations")
    return parser


def save_checkpoint(
    path: Path, model: Autoencoder | VAE, optimizer: torch.optim.Optimizer, *,
    epoch: int, validation: dict[str, float], config: dict,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1, "epoch": epoch, "validation": validation, "config": config,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Autoencoder | VAE, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    config = checkpoint["config"]
    model = build_model(config["kind"], **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    matplotlib.use("Agg")
    for name in ("epochs", "batch_size", "sample_count", "sample_every", "reconstruction_count", "reconstruction_columns"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate",):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("beta", "weight_decay", "grad_clip"):
        if not 0 <= getattr(args, name) < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.model == "ae" and args.beta != 1:
        parser.error("--beta is only meaningful for vae/fvae")
    if args.model != "ae" and args.ae_loss != "mse":
        parser.error("--ae-loss l1 is only meaningful for ae")
    try:
        validate_likelihood(args.likelihood, args.observation_std)
        if not 0 < args.validation_fraction < 1 or args.num_workers < 0:
            raise ValueError("validation fraction must be in (0,1); workers must be nonnegative")
        device = resolve_device(args.device)
        seed_everything(args.seed, deterministic=args.deterministic)
        model_config = {"latent_dim": args.latent_dim, "hidden_channels": args.hidden_channels,
                        "hidden_dim": args.hidden_dim, "conv_depth": args.conv_depth}
        if args.model != "ae":
            model_config.update({
                "flow_steps": args.flow_steps if args.model == "fvae" else 0,
                "flow_hidden_dim": args.flow_hidden_dim, "flow_hidden_layers": args.flow_hidden_layers,
                "scale_limit": args.scale_limit,
            })
        model = build_model(args.model, **model_config).to(device)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or Path("checkpoints") / "vae" / f"{args.model}_{args.dataset}_{uuid4().hex[:8]}"
    # Refuse old runs before any epoch/checkpoint can change them.
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    loaders = make_dataloaders(
        args.dataset, args.data_root, batch_size=args.batch_size, seed=args.seed,
        validation_fraction=args.validation_fraction, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", download=args.download,
    )
    train_loader = limit_loader(loaders.train, args.train_limit, shuffle=True)
    validation_loader = limit_loader(loaders.validation, args.validation_limit, shuffle=False)
    # Dataset indexing avoids consuming the validation DataLoader worker RNG.
    preview = torch.stack([validation_loader.dataset[i][0]
                           for i in range(min(args.reconstruction_count, len(validation_loader.dataset)))])
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    objective = {"likelihood": args.likelihood, "observation_std": args.observation_std, "beta": args.beta}
    variational = isinstance(model, VAE)
    selection_metric = "negative_elbo" if variational else "grayscale_mse"
    config = {
        "kind": args.model, "model": model_config, "arguments": arguments,
        "objective": objective if variational else {
            "ae_loss": args.ae_loss,
            "loss": "sum squared error / image" if args.ae_loss == "mse" else "sum absolute error / image",
        },
        "selection_metric": selection_metric,
        "observation_protocol": ("dynamic binarization; binary encoder inputs and targets"
                                 if variational and args.likelihood == "bernoulli" else "original grayscale [0,1]"),
        "units": ("nats/image; pixel and latent sums, then batch mean" if variational else
                  f"{'SSE' if args.ae_loss == 'mse' else 'SAE'}/image; grayscale_mse is SSE/784; grayscale_mae is SAE/784"),
        "runtime": {"torch_version": str(torch.__version__), "device": str(device),
                    "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_sizes": {"train": len(train_loader.dataset), "validation": len(validation_loader.dataset),
                        "test": len(loaders.test.dataset)},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    print(f"Device: {device}; parameters: {config['parameter_count']:,}; output: {output}", flush=True)
    print(f"Selection: validation {selection_metric}; observations: {config['observation_protocol']}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    options = dict(likelihood=args.likelihood, observation_std=args.observation_std, beta=args.beta)
    if not variational:
        options["ae_loss"] = args.ae_loss
    best_value, best_epoch = float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(
            model, train_loader, device, optimizer=optimizer, seed=args.seed + epoch,
            grad_clip=args.grad_clip, progress=True, **options,
        )
        validation = run_epoch(model, validation_loader, device, seed=args.seed, progress=True, **options)
        row = {"epoch": epoch, "train": train, "validation": validation, "seconds": time.perf_counter() - start}
        history.append(row)
        if validation[selection_metric] < best_value:
            best_value, best_epoch = validation[selection_metric], epoch
            save_checkpoint(output / "best.pt", model, optimizer, epoch=epoch, validation=validation, config=config)
        write_json(output / "history.json", history)
        save_history(history, output, variational=variational)
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_reconstructions(model, preview, output / f"reconstructions_epoch_{epoch:03d}.png",
                                 likelihood=args.likelihood, seed=args.seed,
                                 count=args.reconstruction_count, nrow=args.reconstruction_columns)
            if variational:
                save_samples(model, output / f"samples_epoch_{epoch:03d}.png",
                             seed=args.seed, count=args.sample_count)
        print(f"Epoch {epoch}/{args.epochs}: train objective {train['objective']:.3f}; "
              f"validation {selection_metric} {validation[selection_metric]:.5f}; best epoch {best_epoch}", flush=True)
    best_model, checkpoint = load_checkpoint(output / "best.pt", device)
    save_reconstructions(best_model, preview, output / "reconstructions_best.png",
                         likelihood=args.likelihood, seed=args.seed,
                         count=args.reconstruction_count, nrow=args.reconstruction_columns)
    if variational:
        save_samples(best_model, output / "samples_best.png", seed=args.seed, count=args.sample_count)
        save_samples(best_model, output / "observations_best.png", seed=args.seed, count=args.sample_count,
                     likelihood=args.likelihood, observation_std=args.observation_std, sample_observations=True)
        save_traversal(best_model, output / "traversal_best.png")
    summary = {"best_epoch": checkpoint["epoch"], "selection_metric": selection_metric,
               "validation": checkpoint["validation"]}
    if args.evaluate_test:
        summary["test"] = run_epoch(best_model, loaders.test, device, seed=args.seed + 1, progress=True, **options)
    write_json(output / "summary.json", summary)
    print(f"Finished. Selected checkpoint and figures: {output}", flush=True)


if __name__ == "__main__":
    main()
