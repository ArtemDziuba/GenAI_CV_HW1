"""Freeze the improved classifier and reassess existing weights, without GAN training.

The old and new KID feature spaces have different scales. Historical epochs
were selected under v1; unsaved epochs cannot be reselected with v2.
"""

from datetime import UTC, datetime
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gan.classifier import classification_diagnostics, load_classifier
from gan.evaluate import main as evaluate
from gan.prepare_evaluation import main as prepare
from gan.metrics import polynomial_mmd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
MANIFEST = ROOT / "manifest.json"
CLASSIFIER = Path("checkpoints/gan_evaluation/classifier_fashion_mnist_deep_02")
BUNDLE = Path("data/gan_evaluation/fashion_mnist_v2")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_protected(plan: dict) -> None:
    for name, expected in plan["protected_hashes"].items():
        actual = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Original result or checkpoint changed: {name}")


@torch.no_grad()
def diagnose(path: Path, images: torch.Tensor, labels: torch.Tensor) -> dict:
    model, _ = load_classifier(path, torch.device("cuda"))
    logits = torch.cat([model(batch.cuda())[1].cpu() for batch in images.split(128)])
    return classification_diagnostics(logits, labels)


def main() -> None:
    plan = read(MANIFEST)
    verify_protected(plan)
    selected = read(PROJECT / CLASSIFIER / "summary.json")
    old_summary = read(PROJECT / "checkpoints/gan_evaluation/classifier_fashion_mnist_01/summary.json")
    if selected["validation"]["accuracy"] < .93:
        raise RuntimeError("Classifier has not met the predeclared 93% internal-holdout quality gate")
    plan.update(status="preparing_v2", classifier_summary=selected, previous_classifier_summary=old_summary,
                bundle=BUNDLE.as_posix(), minimum_internal_accuracy=.93,
                metric_note="KID values from v1 and v2 cannot be compared numerically; rank within one feature space only",
                checkpoint_note="Only six v1-selected stored checkpoints reassessed; historical best epochs cannot be reselected")
    write(MANIFEST, plan)
    prepare_args = ["--classifier", str(CLASSIFIER / "best.pt"), "--output", str(BUNDLE), "--count", "5000",
                    "--seed", "2026", "--batch-size", "128", "--device", "cuda", "--min-classifier-accuracy", ".93"]
    prepare(prepare_args)
    old_ref = torch.load(PROJECT / "data/gan_evaluation/fashion_mnist/reference.pt", map_location="cpu", weights_only=True)
    new_ref = torch.load(PROJECT / BUNDLE / "reference.pt", map_location="cpu", weights_only=True)
    for field in ("source_indices", "images", "labels"):
        assert torch.equal(old_ref[field], new_ref[field])
    old_diagnostics = diagnose(PROJECT / "data/gan_evaluation/fashion_mnist/classifier.pt", new_ref["images"], new_ref["labels"])
    new_diagnostics = diagnose(PROJECT / BUNDLE / "classifier.pt", new_ref["images"], new_ref["labels"])
    classifier_comparison = {"old": old_diagnostics, "new": new_diagnostics, "reference_images_identical": True,
                             "new_internal_holdout": selected["validation"], "test_evaluated": False}
    for key, reference in (("old", old_ref), ("new", new_ref)):
        features = reference["features"]
        classifier_comparison[key]["feature_dimension"] = features.shape[1]
        classifier_comparison[key]["mean_feature_l2"] = features.norm(dim=1).mean().item()
        classifier_comparison[key]["real_split_mmd_context_only"] = polynomial_mmd(features[:2500], features[2500:])
    write(ROOT / "classifier_comparison.json", classifier_comparison)
    prior = read(PROJECT / "homeworks/03_gan/artifacts/gan_fashion_width_01/manifest.json")
    rows = []
    plan.update(status="reevaluating", preparation_command=["uv", "run", "--locked", "python", "-m", "gan.prepare_evaluation", *prepare_args], evaluations=[])
    write(MANIFEST, plan)
    for trial in prior["trials"]:
        output = ROOT / "evaluations" / trial["id"]
        args = ["--checkpoint", str(Path(trial["run"]) / "best_kid.pt"), "--evaluation-set", str(BUNDLE),
                "--output", str(output.relative_to(PROJECT)), "--device", "cuda", "--seed", "44", "--sample-seed", "43",
                "--batch-size", "128", "--confidence-threshold", ".8", "--coverage-min-count", "5"]
        evaluate(args)
        result = read(output / "evaluation.json")
        old_inputs = torch.load(PROJECT / trial["run"] / "evaluation_inputs.pt", map_location="cpu", weights_only=True)
        new_inputs = torch.load(output / "evaluation_inputs.pt", map_location="cpu", weights_only=True)
        assert torch.equal(old_inputs["noise"], new_inputs["noise"])
        assert result["checkpoint_epoch"] == trial["best_epoch"] and result["checkpoint_sha256"] == trial["checkpoint_sha256"]
        metric = result["metrics"]
        rows.append({"id": trial["id"], "gc": trial["gc"], "dc": trial["dc"], "stored_checkpoint_epoch": trial["best_epoch"],
                     "v1_kid_separate_scale": trial["kid"], "v2_kid": metric["kid"],
                     "v1_confident_fraction": trial["metrics"]["confident_fraction"], "v2_confident_fraction": metric["confident_fraction"],
                     "v2_mean_confidence": metric["mean_confidence"], "v2_confident_coverage": metric["confident_class_coverage"],
                     "v2_class_counts": metric["class_counts"], "v2_normalized_class_entropy": metric["normalized_class_entropy"]})
        plan["evaluations"].append({"trial": trial["id"], "output": output.relative_to(PROJECT).as_posix(),
                                    "command": ["uv", "run", "--locked", "python", "-m", "gan.evaluate", *args]})
        write(MANIFEST, plan)
    write(ROOT / "metrics.json", rows)
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = read(PROJECT / prior["trials"][0]["run"] / "config.json")["class_names"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    positions = np.arange(10)
    for offset, key in ((-.18, "old"), (.18, "new")):
        diagnostic = classifier_comparison[key]
        axes[0].bar(positions + offset, diagnostic["per_class_recall"], width=.36,
                    label=f"{key}: accuracy {diagnostic['accuracy']:.2%}, ECE {diagnostic['ece']:.3f}")
    axes[0].set_xticks(positions, labels, rotation=55, ha="right")
    axes[0].set(ylabel="Recall on the same 5,000 real reference images", ylim=(0, 1.05))
    axes[0].legend(fontsize=8)
    for position, key in enumerate(("old", "new")):
        diagnostic = classifier_comparison[key]
        axes[1].bar(position - .18, diagnostic["confident_fraction"], width=.36, color="tab:blue", label="Fraction above 0.8" if position == 0 else None)
        axes[1].bar(position + .18, diagnostic["confident_accuracy"], width=.36, color="tab:orange", label="Accuracy within confident subset" if position == 0 else None)
    axes[1].set_xticks([0, 1], ["Original classifier", "Deeper classifier"])
    axes[1].set(ylim=(0, 1.05), title="Confidence diagnostics on labelled real images")
    axes[1].legend(fontsize=8)
    figure.savefig(ROOT / "classifier_quality.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = [f"G{row['gc']}/D{row['dc']}" for row in rows]
    axes[0].bar(names, [row["v2_kid"] for row in rows])
    axes[0].set(title="Six saved checkpoints under the new classifier", ylabel="V2 KID-style MMD (lower is better)")
    positions = np.arange(len(rows))
    for offset, key, label in ((-.18, "v1_confident_fraction", "V1"), (.18, "v2_confident_fraction", "V2")):
        axes[1].bar(positions + offset, [row[key] for row in rows], width=.36, label=label)
    axes[1].set_xticks(positions, names)
    axes[1].set(ylim=(0, 1), ylabel="Generated fraction with confidence >= 0.8", title="Confidence is not generated-image accuracy")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.2)
    figure.savefig(ROOT / "saved_checkpoint_comparison.png", dpi=160)
    plt.close(figure)
    verify_protected(plan)
    plan.update(status="complete", finished_utc=datetime.now(UTC).isoformat(),
                best_stored_checkpoint_v2=min(rows, key=lambda row: row["v2_kid"])["id"],
                classifier_comparison=classifier_comparison)
    write(MANIFEST, plan)
    print("DONE: stronger classifier frozen; six unchanged GAN checkpoints reassessed; no GAN training performed.")


if __name__ == "__main__":
    main()
