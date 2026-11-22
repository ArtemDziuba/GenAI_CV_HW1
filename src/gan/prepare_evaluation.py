"""Cache 5,000 held-out validation images and frozen classifier features once."""

import argparse
from pathlib import Path
import shutil

import torch
from torch.utils.data import Subset

from shared import make_dataloaders
from shared.experiment import resolve_device, write_json
from .classifier import load_classifier
from .evaluation import bundle_id, extract_features, file_hash
from .metrics import class_statistics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh reusable reference bundle")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--count", type=int, default=5000, help="Real AND generated images per checkpoint; other counts are smoke tests")
    parser.add_argument("--seed", type=int, default=2026, help="Fixed reference-subset sampling seed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-classifier-accuracy", type=float, default=.8, help="Reject weak classifiers on reference images; record actual accuracy")
    args = parser.parse_args(argv)
    if args.count < 2 or args.batch_size < 1 or not 0 <= args.min_classifier_accuracy <= 1:
        parser.error("reference count >=2, positive batch size, and accuracy threshold in [0,1] required")
    if not 0 <= args.seed < 2**32:
        parser.error("reference seed must be in [0,2**32)")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh evaluation bundle directory: {args.output}")
    device = resolve_device(args.device)
    classifier, checkpoint = load_classifier(args.classifier, device)
    config = checkpoint["config"]
    loaders = make_dataloaders(config["dataset"], args.data_root, batch_size=args.batch_size,
                              seed=config["split_seed"], validation_fraction=config["validation_fraction"], download=False)
    reference = loaders.validation.dataset
    if args.count > len(reference):
        parser.error("requested reference count exceeds the reserved GAN validation split")
    positions = torch.randperm(len(reference), generator=torch.Generator().manual_seed(args.seed))[:args.count]
    selected = Subset(reference, positions.tolist())
    images, labels = zip(*[selected[index] for index in range(len(selected))])
    images, labels = torch.stack(images), torch.tensor(labels, dtype=torch.long)
    features, probabilities = extract_features(classifier, images, device, args.batch_size)
    accuracy = (probabilities.argmax(1) == labels).float().mean().item()
    if accuracy < args.min_classifier_accuracy:
        raise ValueError(f"Reference classifier accuracy {accuracy:.4f} is below {args.min_classifier_accuracy:.4f}; "
                         "train a better evaluation classifier before selecting GAN checkpoints")
    # Record original official-training indices so split isolation is auditable.
    if not isinstance(reference, Subset):
        raise ValueError("expected the shared seeded validation Subset")
    original_indices = torch.tensor([int(reference.indices[index]) for index in positions])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.classifier, args.output / "classifier.pt")
    torch.save({"images": images, "labels": labels, "features": features, "probabilities": probabilities,
                "source_indices": original_indices}, args.output / "reference.pt")
    metadata = {"kind": "classifier_kid_reference", "format_version": 1, "dataset": config["dataset"],
                "split": "validation", "split_seed": config["split_seed"], "validation_fraction": config["validation_fraction"],
                "reference_seed": args.seed, "count": args.count, "smoke_only": args.count != 5000,
                "real_class_counts": torch.bincount(labels, minlength=10).tolist(),
                "feature_dim": features.shape[1], "classifier_epoch": checkpoint["epoch"],
                "classifier_reference_accuracy": accuracy, "min_classifier_accuracy": args.min_classifier_accuracy,
                "preprocessing": config["preprocessing"], "classifier_fit_pool": config["fit_pool"],
                "source": "official training images held out by shared validation split; official test unused",
                "reference_diagnostics": class_statistics(probabilities),
                "file_hashes": {filename: file_hash(args.output / filename) for filename in ("classifier.pt", "reference.pt")}}
    metadata["bundle_id"] = bundle_id(metadata)
    write_json(args.output / "config.json", metadata)
    print(f"Prepared {args.count} {config['dataset']} reference images; classifier accuracy {accuracy:.4f}; "
          f"bundle {metadata['bundle_id']}; no test evaluation")


if __name__ == "__main__":
    main()
