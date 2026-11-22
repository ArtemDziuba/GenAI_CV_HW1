"""Evaluate an existing GAN checkpoint against a prepared frozen reference bundle."""

import argparse
from pathlib import Path

import matplotlib
import torch

from shared.experiment import resolve_device, write_json
from .artifacts import fixed_inputs, save_samples
from .evaluation import GeneratorEvaluator, file_hash
from .train import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=.8)
    parser.add_argument("--coverage-min-count", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=43)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Use a fresh evaluation directory: {args.output}")
    matplotlib.use("Agg")
    device = resolve_device(args.device)
    generator, _, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    evaluator = GeneratorEvaluator(args.evaluation_set, device, latent_dim=generator.latent_dim,
                                   conditional=generator.conditional, seed=args.seed, batch_size=args.batch_size,
                                   confidence_threshold=args.confidence_threshold, coverage_min_count=args.coverage_min_count)
    split_seed = config.get("data_split_seed", config["arguments"]["seed"])
    if (config["arguments"]["dataset"] != evaluator.config["dataset"] or split_seed != evaluator.config["split_seed"]
            or config["arguments"]["validation_fraction"] != evaluator.config["validation_fraction"]):
        raise ValueError("reference dataset/split differs from checkpoint training; reference images might overlap training")
    digest = file_hash(args.checkpoint)
    metrics = evaluator.evaluate(generator)
    args.output.mkdir(parents=True, exist_ok=False)
    evaluator.save_inputs(args.output / "evaluation_inputs.pt")
    noise, labels = fixed_inputs(generator, count=args.sample_count, samples_per_class=args.samples_per_class, seed=args.sample_seed)
    save_samples(generator, noise, labels, args.output / "samples.png", class_names=config["class_names"],
                 columns=args.samples_per_class if generator.conditional else 8)
    torch.save({"noise": noise, "labels": labels}, args.output / "preview_inputs.pt")
    if file_hash(args.checkpoint) != digest:
        raise RuntimeError("checkpoint changed during evaluation")
    write_json(args.output / "evaluation.json", {"checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
                                                "checkpoint_epoch": checkpoint["epoch"], "metrics": metrics,
                                                "protocol": evaluator.protocol, "preview_seed": args.sample_seed,
                                                "split": "validation reference; not test", "training_performed": False})
    print(f"KID-style MMD {metrics['kid']:.6f}; confident coverage {metrics['confident_class_coverage']}/10; "
          f"mean confidence {metrics['mean_confidence']:.4f}; saved {args.output}")


if __name__ == "__main__":
    main()
