"""Train Task 3 variants: python -m gan.train --help.

Non-saturating G loss; D loss averages real/fake BCE. Feature matching adds a
batch-mean feature MSE. Select one best checkpoint using frozen classifier-feature
KID-style MMD on 5,000 validation reference images and 5,000 fixed-noise generations.

--resume restores paired G/D and Adam states into a fresh output directory.
--epochs is the final total epoch, not the number of additional epochs.
Legacy checkpoints without RNG snapshots resume valid training branches;
their earlier sampling/shuffle trajectory cannot be reproduced exactly.
"""

import argparse
import csv
from pathlib import Path
import time
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import torch

from shared import make_dataloaders, plot_image_grid, seed_everything
from shared.experiment import limit_loader, resolve_device, write_json
from .artifacts import fixed_inputs, save_evaluation_history, save_history, save_samples
from .evaluation import GeneratorEvaluator, file_hash
from .models import Discriminator, Generator
from .training import run_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist"), default="mnist")
    parser.add_argument("--evaluation-set", type=Path, required=True, help="Frozen bundle from gan.prepare_evaluation; reused across runs")
    parser.add_argument("--resume", type=Path, help="Restore paired G/D and optimizers; inherit saved settings unless explicitly overridden")
    parser.add_argument("--evaluation-seed", type=int, default=44, help="Fixed 5,000-vector evaluation noise bank seed")
    parser.add_argument("--evaluation-batch-size", type=int, default=128, help="Classifier and generator evaluation batch size")
    parser.add_argument("--kid-block-size", type=int, default=512, help="CPU kernel block size; does not subsample the reference")
    parser.add_argument("--confidence-threshold", type=float, default=.8, help="Threshold for confidence-filtered coverage/entropy")
    parser.add_argument("--coverage-min-count", type=int, default=5, help="Confident samples required to count a covered class")
    parser.add_argument("--conditional", action="store_true", help="Condition both G and D on ten class labels")
    parser.add_argument("--discriminator", choices=("global", "patch"), default="global", help="One image logit or 7x7 patch logits")
    parser.add_argument("--feature-matching-weight", type=float, default=0., help="Add lambda times batch-mean feature MSE to G loss; 0 disables")
    parser.add_argument("--latent-dim", type=int, default=100, help="Number of standard Gaussian noise coordinates")
    parser.add_argument("--generator-channels", type=int, default=64, help="G uses 2*C then C feature channels")
    parser.add_argument("--discriminator-channels", type=int, default=64, help="D uses C then 2*C feature channels")
    parser.add_argument("--embedding-dim", type=int, default=32, help="G class embedding width, used only with --conditional")
    parser.add_argument("--epochs", type=int, default=30, help="Final total epoch; with --resume this includes the saved epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Real/fake images per optimization minibatch")
    parser.add_argument("--generator-lr", type=float, default=0.0002, help="Adam learning rate for G")
    parser.add_argument("--discriminator-lr", type=float, default=0.0002, help="Adam learning rate for D")
    parser.add_argument("--adam-beta1", type=float, default=0.5, help="First-moment decay, shared by both optimizers")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Second-moment decay, shared by both optimizers")
    parser.add_argument("--discriminator-steps", type=int, default=1, help="D updates per G update, with fresh noise each time")
    parser.add_argument("--sample-every", type=int, default=5, help="Save preview grids; also save epoch 1 and final epoch")
    parser.add_argument("--sample-count", type=int, default=64, help="Unconditional preview image count")
    parser.add_argument("--samples-per-class", type=int, default=8, help="Conditional preview columns: ten labelled rows")
    parser.add_argument("--sample-seed", type=int, default=43, help="Independent fixed preview noise seed")
    parser.add_argument("--seed", type=int, default=42, help="Model initialization and training noise; evaluation bundle fixes dataset split")
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch algorithms; may reject unsupported CUDA ops")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of official training data reserved for diagnostics")
    parser.add_argument("--num-workers", type=int, default=0, help="Data-loader worker processes; 0 is convenient on Windows")
    parser.add_argument("--train-limit", type=int, help="Subset smoke test only")
    parser.add_argument("--validation-limit", type=int, help="Subset smoke test only")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Explicitly permit torchvision dataset download")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="Fresh directory; existing runs are refused")
    return parser


def save_checkpoint(path: Path, generator: Generator, discriminator: Discriminator,
                    generator_optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer,
                    *, epoch: int, config: dict, validation: dict, selection: dict | None = None,
                    training_state: dict | None = None) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "epoch": epoch, "config": config, "validation": validation, "selection": selection,
                "generator_state": generator.state_dict(), "discriminator_state": discriminator.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict(), "training_state": training_state}, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Generator, Discriminator, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("unsupported GAN checkpoint format")
    config = checkpoint["config"]
    generator = Generator(**config["generator"]).to(device)
    discriminator = Discriminator(**config["discriminator"]).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    return generator.eval(), discriminator.eval(), checkpoint


def capture_training_state(train_loader, validation_loader, device: torch.device, global_step: int) -> dict:
    """Torch and loader RNGs used by this trainer, saved at an epoch boundary."""
    return {"global_g_step": global_step, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "train_loader_rng": train_loader.generator.get_state() if train_loader.generator is not None else None,
            "validation_loader_rng": validation_loader.generator.get_state() if validation_loader.generator is not None else None}


def restore_training_state(state: dict | None, train_loader, validation_loader, device: torch.device) -> dict:
    """Old checkpoints may omit RNGs; report which states were restored."""
    restored = dict.fromkeys(("torch_rng", "cuda_rng", "train_loader_rng", "validation_loader_rng"), False)
    if not state:
        return restored
    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"].cpu())
        restored["torch_rng"] = True
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        if len(state["cuda_rng"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
            restored["cuda_rng"] = True
    for key, loader in (("train_loader_rng", train_loader), ("validation_loader_rng", validation_loader)):
        if state.get(key) is not None and loader.generator is not None:
            loader.generator.set_state(state[key].cpu())
            restored[key] = True
    return restored


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--resume", type=Path)
    resume_path = preliminary.parse_known_args(argv)[0].resume
    resumed = None
    resume_digest = None
    if resume_path is not None:
        resume_digest = file_hash(resume_path)
        resumed = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (resumed.get("format_version") != 1 or not resumed.get("generator_optimizer")
                or not resumed.get("discriminator_optimizer")):
            parser.error("resume requires a paired GAN checkpoint with both optimizer states")
        excluded = {"output", "epochs", "resume", "device", "download", "evaluation_set"}
        parser.set_defaults(**{key: value for key, value in resumed["config"]["arguments"].items() if key not in excluded})
    args = parser.parse_args(argv)
    start_epoch = int(resumed["epoch"]) if resumed is not None else 0
    if resumed is not None and args.epochs <= start_epoch:
        parser.error("--epochs must exceed the saved epoch; it specifies the final total epoch")
    positive = ("latent_dim", "generator_channels", "discriminator_channels", "embedding_dim", "epochs", "batch_size",
                "discriminator_steps", "sample_every", "sample_count", "samples_per_class",
                "evaluation_batch_size", "kid_block_size", "coverage_min_count")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("model dimensions, epochs, counts and update intervals must be positive")
    if not all(0 < value < float("inf") for value in (args.generator_lr, args.discriminator_lr)):
        parser.error("learning rates must be finite and positive")
    if not all(0 <= value < 1 for value in (args.adam_beta1, args.adam_beta2)):
        parser.error("Adam beta values must be in [0,1)")
    if not 0 <= args.feature_matching_weight < float("inf"):
        parser.error("feature matching weight must be finite and nonnegative")
    if not 0 < args.validation_fraction < 1 or args.num_workers < 0:
        parser.error("validation fraction must be in (0,1); workers must be nonnegative")
    if not all(0 <= value < 2**32 for value in (args.seed, args.sample_seed)):
        parser.error("seeds must be in [0, 2**32)")
    output = args.output or Path("checkpoints/gan") / f"{'cgan' if args.conditional else 'gan'}_{args.dataset}_{uuid4().hex[:8]}"
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    matplotlib.use("Agg")
    device = resolve_device(args.device)
    evaluator = GeneratorEvaluator(args.evaluation_set, device, latent_dim=args.latent_dim, conditional=args.conditional,
                                   seed=args.evaluation_seed, batch_size=args.evaluation_batch_size,
                                   confidence_threshold=args.confidence_threshold, coverage_min_count=args.coverage_min_count,
                                   block_size=args.kid_block_size)
    if evaluator.config["dataset"] != args.dataset or evaluator.config["validation_fraction"] != args.validation_fraction:
        parser.error("evaluation bundle dataset/validation fraction differs from the training configuration")
    if evaluator.count != 5000 and args.train_limit is None:
        parser.error("production checkpoint selection needs 5,000 references; small bundles require an explicit --train-limit smoke run")
    if resumed is not None:
        previous = resumed["config"]
        if previous.get("evaluation_protocol") != evaluator.protocol:
            parser.error("resume requires the same frozen evaluation protocol and noise bank")
        unchanged = ("dataset", "validation_fraction", "batch_size", "train_limit", "validation_limit", "seed",
                     "adam_beta1", "adam_beta2", "discriminator_steps", "feature_matching_weight",
                     "sample_seed", "sample_count", "samples_per_class")
        for key in unchanged:
            if previous["arguments"].get(key) != getattr(args, key):
                parser.error(f"resume cannot change {key}; only learning-rate overrides are supported")
    # Classifier construction/loading must not perturb G/D initialization RNGs.
    seed_everything(args.seed, deterministic=args.deterministic)
    generator_config = dict(latent_dim=args.latent_dim, channels=args.generator_channels,
                            conditional=args.conditional, embedding_dim=args.embedding_dim, num_classes=10)
    discriminator_config = dict(channels=args.discriminator_channels, conditional=args.conditional,
                                head=args.discriminator, num_classes=10)
    if resumed is not None and (generator_config != resumed["config"]["generator"]
                                or discriminator_config != resumed["config"]["discriminator"]):
        parser.error("resume requires matching generator and discriminator architectures")
    generator = Generator(**generator_config).to(device)
    discriminator = Discriminator(**discriminator_config).to(device)
    if resumed is not None:
        generator.load_state_dict(resumed["generator_state"])
        discriminator.load_state_dict(resumed["discriminator_state"])
    loaders = make_dataloaders(args.dataset, args.data_root, batch_size=args.batch_size, seed=evaluator.config["split_seed"],
                              validation_fraction=args.validation_fraction, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", download=args.download)
    train_loader = limit_loader(loaders.train, args.train_limit, shuffle=True)
    validation_loader = limit_loader(loaders.validation, args.validation_limit, shuffle=False)
    optimizer_options = dict(betas=(args.adam_beta1, args.adam_beta2))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr, **optimizer_options)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, **optimizer_options)
    if resumed is not None:
        current_sizes = {"train": len(train_loader.dataset), "validation": len(validation_loader.dataset), "test": len(loaders.test.dataset)}
        if resumed["config"]["split_sizes"] != current_sizes:
            parser.error("resume dataset sizes differ from the saved training configuration")
        g_optimizer.load_state_dict(resumed["generator_optimizer"])
        d_optimizer.load_state_dict(resumed["discriminator_optimizer"])
        # Loading Adam restores saved param-group LRs; apply any explicit overrides afterwards.
        for optimizer, rate in ((g_optimizer, args.generator_lr), (d_optimizer, args.discriminator_lr)):
            for group in optimizer.param_groups:
                group["lr"] = rate
    initial_step = (resumed.get("training_state") or {}).get("global_g_step", start_epoch * len(train_loader)) if resumed else 0
    # Follow nested Subset wrappers without duplicating dataset-specific label names.
    dataset = validation_loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    class_names = list(getattr(dataset, "classes", [str(i) for i in range(10)]))
    config = {"generator": generator_config, "discriminator": discriminator_config,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "parameter_count": {"generator": sum(p.numel() for p in generator.parameters()),
                                  "discriminator": sum(p.numel() for p in discriminator.parameters())},
              "split_sizes": {"train": len(train_loader.dataset), "validation": len(validation_loader.dataset), "test": len(loaders.test.dataset)},
              "class_names": class_names, "preprocessing": "original grayscale [0,1] mapped to [-1,1]; no binarization",
              "objectives": {"discriminator": "0.5 * (mean softplus(-D(real)) + mean softplus(D(fake)))",
                             "generator": "mean softplus(-D(fake)) + feature_matching_weight * feature_MSE",
                             "feature_matching_weight": args.feature_matching_weight,
                             "feature_MSE": "MSE of batch-mean penultimate features, averaged over channels/spatial positions",
                             "patch_reduction": "BCE averaged over images and patches"},
              "evaluation_protocol": evaluator.protocol,
              "data_split_seed": evaluator.config["split_seed"],
              "checkpoint_selection": {"split": "validation", "criteria": {"kid": "kid"},
                                       "files": ["best_kid.pt"],
                                       "ties": "keep earliest epoch", "snapshots": "paired G/D weights and both optimizers at the selected epoch"},
              "runtime": {"torch": str(torch.__version__), "device": str(device), "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    if resumed is not None:
        restored_rngs = restore_training_state(resumed.get("training_state"), train_loader, validation_loader, device)
        config["resumption"] = {"checkpoint": str(args.resume), "checkpoint_sha256": resume_digest,
                                "saved_epoch": start_epoch, "first_training_epoch": start_epoch + 1,
                                "target_total_epoch": args.epochs, "optimizer_states_restored": True,
                                "restored_rngs": restored_rngs, "initial_g_updates": initial_step,
                                "history_policy": "Only continued epochs are logged here; source history remains separate",
                                "reproducibility": "Legacy checkpoints without RNG snapshots branch from saved weights/optimizers; CUDA may be nondeterministic"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    evaluator.save_inputs(output / "evaluation_inputs.pt")
    write_json(output / "evaluation_protocol.json", {**evaluator.protocol,
                                                      "fixed_inputs_sha256": file_hash(output / "evaluation_inputs.pt")})
    preview_noise, preview_labels = fixed_inputs(generator, count=args.sample_count,
                                               samples_per_class=args.samples_per_class, seed=args.sample_seed)
    torch.save({"noise": preview_noise, "labels": preview_labels}, output / "preview_inputs.pt")
    columns = args.samples_per_class if args.conditional else 8
    save_samples(generator, preview_noise, preview_labels, output / f"samples_epoch_{start_epoch:03d}.png", class_names=class_names, columns=columns)
    preview = torch.stack([validation_loader.dataset[index][0] for index in range(min(64, len(validation_loader.dataset)))])
    figure = plot_image_grid(preview, max_images=64, save_path=output / "real_validation.png", title="Real validation images")
    plt.close(figure)
    print(f"Device {device}; parameters {config['parameter_count']}; output {output}", flush=True)
    print(f"Selecting minimum KID-style MMD: {evaluator.count} fixed real and generated images; frozen classifier.", flush=True)
    history, sampled_epochs, best_checkpoints, global_step = [], [], {}, initial_step
    baseline = None
    if resumed is not None:
        # Compare continued training with the loaded model, so regressions cannot
        # discard the starting best checkpoint in this new output directory.
        evaluation = evaluator.evaluate(generator)
        baseline = {"epoch": start_epoch, "train": None, "validation": resumed["validation"], "evaluation": evaluation}
        record = {"epoch": start_epoch, "score": evaluation["kid"], "criterion": "kid", "evaluation": evaluation,
                  "split": "validation", "checkpoint": "best_kid.pt", "sample_grid": "samples_best_kid.png",
                  "from_resume": True}
        record["preview"] = save_samples(generator, preview_noise, preview_labels, output / record["sample_grid"], class_names=class_names, columns=columns)
        save_checkpoint(output / "best_kid.pt", generator, discriminator, g_optimizer, d_optimizer, epoch=start_epoch,
                        config=config, validation=resumed["validation"], selection=record,
                        training_state=capture_training_state(train_loader, validation_loader, device, global_step))
        best_checkpoints["kid"] = record
        sampled_epochs.append(start_epoch)
        write_json(output / "resume_baseline.json", baseline)
        write_json(output / "best_checkpoints.json", best_checkpoints)
        if file_hash(args.resume) != resume_digest:
            raise RuntimeError("source checkpoint changed during resume initialization")
        print(f"Resumed epoch {start_epoch}; training {start_epoch + 1}..{args.epochs}; starting KID {evaluation['kid']:.6f}; RNG restoration {config['resumption']['restored_rngs']}", flush=True)
    with (output / "steps.csv").open("w", encoding="utf-8", newline="", buffering=1) as stream:
        writer = None
        for epoch in range(start_epoch + 1, args.epochs + 1):
            started = time.perf_counter()
            def log_step(batch_index, metrics):
                nonlocal writer, global_step
                global_step += 1
                row = {"epoch": epoch, "batch": batch_index, "global_g_step": global_step,
                       "global_d_step": global_step * args.discriminator_steps, **metrics}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            options = dict(discriminator_steps=args.discriminator_steps, feature_matching_weight=args.feature_matching_weight, progress=True)
            train = run_epoch(generator, discriminator, train_loader, device, generator_optimizer=g_optimizer,
                              discriminator_optimizer=d_optimizer, seed=(args.seed + epoch) % 2**32, on_step=log_step, **options)
            validation = run_epoch(generator, discriminator, validation_loader, device, seed=args.seed, **options)
            evaluation = evaluator.evaluate(generator)
            row = {"epoch": epoch, "train": train, "validation": validation, "evaluation": evaluation,
                   "seconds": time.perf_counter() - started}
            if epoch == start_epoch + 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
                row["preview"] = save_samples(generator, preview_noise, preview_labels, output / f"samples_epoch_{epoch:03d}.png",
                                              class_names=class_names, columns=columns)
                sampled_epochs.append(epoch)
            if "kid" not in best_checkpoints or evaluation["kid"] < best_checkpoints["kid"]["score"]:
                record = {"epoch": epoch, "score": evaluation["kid"], "criterion": "kid", "evaluation": evaluation,
                          "split": "validation", "checkpoint": "best_kid.pt", "sample_grid": "samples_best_kid.png"}
                record["preview"] = save_samples(generator, preview_noise, preview_labels, output / record["sample_grid"],
                                                 class_names=class_names, columns=columns)
                save_checkpoint(output / record["checkpoint"], generator, discriminator, g_optimizer, d_optimizer,
                                epoch=epoch, config=config, validation=validation, selection=record,
                                training_state=capture_training_state(train_loader, validation_loader, device, global_step))
                best_checkpoints["kid"] = record
            write_json(output / "best_checkpoints.json", best_checkpoints)
            history.append(row)
            write_json(output / "history.json", history)
            save_history(history, output)
            save_evaluation_history(history, output)
            print(f"Epoch {epoch}/{args.epochs}: G {train['g_loss']:.4f}, D {train['d_loss']:.4f}; "
                  f"validation G {validation['g_loss']:.4f}, D {validation['d_loss']:.4f}; "
                  f"KID {evaluation['kid']:.6f}, confident coverage {evaluation['confident_class_coverage']}/10, "
                  f"confidence {evaluation['mean_confidence']:.3f}; best KID epoch {best_checkpoints['kid']['epoch']}", flush=True)
    save_samples(generator, preview_noise, preview_labels, output / "samples_last.png", class_names=class_names, columns=columns)
    selected = next((row for row in history if row["epoch"] == best_checkpoints["kid"]["epoch"]), baseline)
    write_json(output / "summary.json", {"last_epoch": args.epochs, "checkpoint": "best_kid.pt",
                                        "best_epoch": selected["epoch"], "selection_metric": "kid",
                                        "best_checkpoints": best_checkpoints,
                                        "train": selected["train"], "validation": selected["validation"],
                                        "evaluation": selected["evaluation"], "last_evaluation": history[-1]["evaluation"],
                                        "last_train": history[-1]["train"], "last_validation": history[-1]["validation"],
                                        "sampled_epochs": sampled_epochs, "g_updates": global_step,
                                        "d_updates": global_step * args.discriminator_steps,
                                        "start_epoch": start_epoch, "first_training_epoch": start_epoch + 1,
                                        "epochs_trained_this_run": args.epochs - start_epoch,
                                        "session_g_updates": global_step - initial_step,
                                        "session_d_updates": (global_step - initial_step) * args.discriminator_steps,
                                        "note": "Minimum frozen classifier-feature KID-style MMD; not standard Inception KID. No test evaluation."})
    print(f"Finished: {output}", flush=True)


if __name__ == "__main__":
    main()
