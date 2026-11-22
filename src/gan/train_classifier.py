"""Train an evaluation classifier separately; never fit on GAN validation/test images.

The GAN's 54,000-image training pool is split internally into classifier fitting
and classifier checkpoint-selection sets. The GAN's 6,000 validation images are
reserved for reference features, and the official test set is not evaluated.
"""

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from shared import make_dataloaders, seed_everything
from shared.experiment import limit_loader, resolve_device, write_json
from .classifier import EvaluationClassifier, classification_diagnostics


class TrainingAugmentation(Dataset):
    """Mild spatial variation on fitting images only; validation stays clean."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip()])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return self.transform(image), label


def classifier_epoch(model: EvaluationClassifier, loader: DataLoader, device: torch.device,
                     optimizer: torch.optim.Optimizer | None = None) -> dict:
    model.train(optimizer is not None)
    loss_sum, correct, count = 0., 0, 0
    saved_logits, saved_labels = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            _, logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite classifier loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(images)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(images)
            if optimizer is None:
                saved_logits.append(logits.cpu())
                saved_labels.append(labels.cpu())
    if not count:
        raise ValueError("empty classifier loader")
    metrics = {"cross_entropy": loss_sum / count, "accuracy": correct / count, "images": count}
    if optimizer is None:
        metrics["diagnostics"] = classification_diagnostics(torch.cat(saved_logits), torch.cat(saved_labels))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--architecture", choices=("simple", "deep"), default="simple")
    parser.add_argument("--dropout", type=float, default=.25, help="Deep classifier only")
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--min-learning-rate", type=float, default=.00001, help="Cosine schedule endpoint")
    parser.add_argument("--augment", action="store_true", help="Fitting only: random crop with padding 2 and horizontal flip")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.1, help="GAN's reserved validation fraction")
    parser.add_argument("--classifier-validation-fraction", type=float, default=.1, help="Internal fraction of GAN training pool")
    parser.add_argument("--train-limit", type=int, help="Classifier smoke test only")
    parser.add_argument("--validation-limit", type=int, help="Internal classifier validation smoke test only")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.channels, args.feature_dim) < 1 or not 0 < args.learning_rate < math.inf:
        parser.error("dimensions/epochs/batch size and learning rate must be positive and finite")
    if not 0 < args.classifier_validation_fraction < 1:
        parser.error("internal classifier validation fraction must be in (0,1)")
    if not 0 <= args.dropout < 1 or not 0 <= args.weight_decay < math.inf or args.num_workers < 0:
        parser.error("dropout in [0,1), finite nonnegative weight decay and nonnegative workers required")
    if args.scheduler == "cosine" and not 0 < args.min_learning_rate <= args.learning_rate:
        parser.error("cosine minimum learning rate must be positive and <= initial learning rate")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh classifier directory: {args.output}")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    loaders = make_dataloaders(args.dataset, args.data_root, batch_size=args.batch_size,
                              seed=args.split_seed, validation_fraction=args.validation_fraction, download=args.download)
    pool = loaders.train.dataset
    held = int(len(pool) * args.classifier_validation_fraction)
    if held < 1 or held >= len(pool):
        parser.error("internal classifier validation must leave nonempty splits")
    fitting, validation = random_split(pool, [len(pool) - held, held], generator=torch.Generator().manual_seed(args.seed))
    if args.augment:
        fitting = TrainingAugmentation(fitting)
    loader_options = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.num_workers > 0)
    fitting_loader = DataLoader(fitting, batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed), **loader_options)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, **loader_options)
    fitting_loader = limit_loader(fitting_loader, args.train_limit, shuffle=True)
    validation_loader = limit_loader(validation_loader, args.validation_limit, shuffle=False)
    model_config = {"channels": args.channels, "feature_dim": args.feature_dim}
    if args.architecture == "deep":
        model_config.update(architecture="deep", dropout=args.dropout)
    model = EvaluationClassifier(**model_config).to(device)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)
                 if args.scheduler == "cosine" else None)
    config = {"dataset": args.dataset, "model": model_config, "split_seed": args.split_seed,
              "validation_fraction": args.validation_fraction, "seed": args.seed,
              "preprocessing": "original grayscale [0,1]; no binarization or per-image normalization",
              "fit_pool": "GAN training split only; internal validation also within that pool",
              "fit_images": len(fitting_loader.dataset), "selection_images": len(validation_loader.dataset),
              "reserved_gan_validation_images": len(loaders.validation.dataset),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)
    history, best_loss, best_epoch = [], math.inf, 0
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train = classifier_epoch(model, fitting_loader, device, optimizer)
        validation_metrics = classifier_epoch(model, validation_loader, device)
        history.append({"epoch": epoch, "learning_rate": learning_rate, "train": train, "validation": validation_metrics})
        if validation_metrics["cross_entropy"] < best_loss:
            best_loss, best_epoch = validation_metrics["cross_entropy"], epoch
            temporary = args.output / "best.tmp"
            torch.save({"kind": "evaluation_classifier", "format_version": 1, "epoch": epoch,
                        "config": config, "validation": validation_metrics, "model_state": model.state_dict()}, temporary)
            temporary.replace(args.output / "best.pt")
        write_json(args.output / "history.json", history)
        print(f"Classifier epoch {epoch}: train accuracy {train['accuracy']:.4f}, internal validation accuracy "
              f"{validation_metrics['accuracy']:.4f}; selected epoch {best_epoch}", flush=True)
        if scheduler is not None:
            scheduler.step()
    write_json(args.output / "summary.json", {"best_epoch": best_epoch, "selection_metric": "internal validation cross entropy",
                                            "validation": history[best_epoch - 1]["validation"], "test_evaluated": False})


if __name__ == "__main__":
    main()
