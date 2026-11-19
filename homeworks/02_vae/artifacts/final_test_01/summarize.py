"""Publish the two frozen final-test comparisons from saved evaluations only."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(row) + " |" for row in rows]])


def number(metrics: dict, key: str, digits: int = 4) -> str:
    return f"{metrics[key]:.{digits}f}" if key in metrics else "n/a"


def architecture(selection: dict) -> str:
    model = selection["model"]
    text = "/".join(str(model[key]) for key in ("latent_dim", "hidden_channels", "hidden_dim", "conv_depth"))
    if selection["kind"] == "fvae":
        text += f"; F{model['flow_steps']}/FH{model['flow_hidden_dim']}"
    return text


def main() -> None:
    plan = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert plan["status"] == "complete" and plan["selection_frozen"]
    records = []
    for selection in plan["selections"]:
        assert selection["status"] == "complete"
        result = json.loads((ROOT / selection["output"] / "evaluation.json").read_text(encoding="utf-8"))
        assert result["count"] == 10000 and result["split"] == "test"
        assert result["metrics"] == selection["test"]
        assert hashlib.sha256((PROJECT_ROOT / selection["run"] / "best.pt").read_bytes()).hexdigest() == selection["checkpoint_sha256"]
        records.append({"dataset": selection["dataset"], "kind": selection["kind"], "run": selection["run"],
                        "architecture": architecture(selection), "parameters": selection["parameters"],
                        "checkpoint_epoch": selection["checkpoint_epoch"], "checkpoint_sha256": selection["checkpoint_sha256"],
                        "test_count": result["count"], "seed": result["seed"],
                        **{"validation_" + key: value for key, value in selection["validation"].items()},
                        **{"test_" + key: value for key, value in selection["test"].items()}})
    (ROOT / "metrics.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (ROOT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Task 2 final test evaluation", "",
             "Both requested steps are complete. Six frozen checkpoints were evaluated once each on the full official "
             "10,000-image test split of their dataset. No training, downloads, or original checkpoint changes occurred.", "",
             "## Selection and protocol", "",
             "Step 1 selects the lowest saved full-split validation MSE for AE, and the lowest saved Bernoulli "
             "validation ordinary negative ELBO for VAE and f-VAE. Test metrics were not used to select checkpoints. "
             "The variational selections all have beta=1.", "",
             "Step 2 matches each model family between Digits and Fashion-MNIST. The six step 1 winners already "
             "form three exact architecture/objective pairs, so this step reuses those evaluations rather than "
             "resampling or testing additional models.", "",
             "Evaluation uses CUDA, batch 64, seed 42, original grayscale AE targets, and the recorded Bernoulli "
             "binarization protocol for VAE/f-VAE. Variational results include a 32-sample importance-weighted bound. "
             "Likelihood terms are nats/image and MSE is per pixel. Negative ELBO/IWAE are estimated bounds, not exact "
             "marginal NLL. Their separate passes use seeded stochastic observation/posterior draws. Finite-sample "
             "estimates need not obey a strict sample-wise ordering.", ""]
    report_lines = ["### Final test evaluation", "",
                    "Both steps use the full official 10,000-image test split per dataset. Checkpoints were frozen "
                    "using validation results before evaluation. CUDA, batch 64, seed 42; variational models use "
                    "Bernoulli observations, beta=1 and an additional IWAE bound with 32 importance samples. "
                    "Likelihood terms are nats/image; MSE is per pixel. No retraining was performed.", "",
                    "#### Step 1: top validation performers", ""]
    for dataset, title in (("mnist", "MNIST Digits"), ("fashion_mnist", "Fashion-MNIST")):
        group = sorted((selection for selection in plan["selections"] if selection["dataset"] == dataset),
                       key=lambda item: ("ae", "vae", "fvae").index(item["kind"]))
        rows = [[item["kind"].upper(), architecture(item), str(item["checkpoint_epoch"]),
                 number(item["validation"], "grayscale_mse", 6), number(item["test"], "grayscale_mse", 6),
                 number(item["validation"], "negative_elbo"), number(item["test"], "negative_elbo"),
                 number(item["test"], "negative_iwae_bound")] for item in group]
        section = [f"**{title}**", "", table(["Model", "Z/C/H/D", "Epoch", "Val MSE", "Test MSE",
                                                    "Val negative ELBO", "Test negative ELBO", "Test negative IWAE-32"], rows), ""]
        lines.extend([f"## Step 1: {title}", "", *section[2:]])
        report_lines.extend(section)
    paired_rows = []
    for kind in ("ae", "vae", "fvae"):
        left = next(item for item in plan["selections"] if item["kind"] == kind and item["dataset"] == "mnist")
        right = next(item for item in plan["selections"] if item["kind"] == kind and item["dataset"] == "fashion_mnist")
        assert left["model"] == right["model"] and left["objective"] == right["objective"]
        assert left["parameters"] == right["parameters"]
        paired_rows.append([kind.upper(), architecture(left), f"{left['parameters']:,}",
                            number(left["test"], "grayscale_mse", 6), number(right["test"], "grayscale_mse", 6),
                            number(left["test"], "negative_elbo"), number(right["test"], "negative_elbo")])
    paired = table(["Model", "Identical Z/C/H/D", "Parameters per model", "Digits test MSE", "Fashion test MSE",
                    "Digits test negative ELBO", "Fashion test negative ELBO"], paired_rows)
    pair_section = ["Step 1 already supplies exact cross-dataset architecture matches, including the same likelihood, "
                    "beta and flow settings. These rows reuse the same six evaluations.", "", paired, ""]
    lines.extend(["## Step 2: matched architectures between datasets", "", *pair_section])
    report_lines.extend(["#### Step 2: matched architectures between datasets", "", *pair_section,
                         "Negative ELBO and IWAE are likelihood-bound estimates. AE and variational MSE also differ "
                         "in observation protocol and deterministic versus sampled reconstruction, so their values "
                         "do not isolate architecture effects. These test results do not measure generation diversity "
                         "or disentanglement.", "",
                         "[Full test results, figures and provenance](artifacts/final_test_01/results.md).", ""])
    gaps = []
    for item in plan["selections"]:
        if item["kind"] != "ae":
            delta = item["test"]["negative_elbo"] - item["validation"]["negative_elbo"]
            gaps.append(f"{item['dataset']} {item['kind'].upper()}: {delta:+.4f}")
    interpretation = ("Test minus validation negative-ELBO gaps (nats/image) are " + "; ".join(gaps) + ". "
                      "The AE test MSEs are also close to their validation values. "
                      "On Digits the selected f-VAE has a lower test bound than the selected VAE; on Fashion "
                      "the selected VAE is slightly lower. The image backbones differ between these model "
                      "families, and only one seed is evaluated, so this does not establish a general flow advantage. "
                      "Step 2 matches backbones across datasets within each family, not across AE/VAE/f-VAE.")
    lines.extend(["## Observations", "", interpretation, ""])
    report_lines.extend([interpretation, ""])
    lines.extend(["## Interpretation limits", "",
                  "These are one training seed and one evaluation seed per configuration. Small differences are "
                  "not statistical rankings. AE uses grayscale inputs and deterministic reconstruction; the "
                  "variational models use binarized inputs and sampled posterior reconstructions. Their MSE "
                  "differences therefore do not isolate architecture. These metrics do not directly measure prior "
                  "generation quality, diversity or disentanglement. No additional tuning follows from test scores.", "",
                  "## Individual artifacts", ""])
    for item in plan["selections"]:
        output = item["output"]
        links = f"[metrics]({output}/evaluation.json), [24 test reconstructions]({output}/reconstructions.png)"
        if item["kind"] != "ae":
            links += f", [prior decoder means]({output}/samples.png), [observation draws]({output}/observations.png), [prior traversal]({output}/traversal.png)"
        lines.append(f"- {item['dataset']} {item['kind'].upper()}, checkpoint epoch {item['checkpoint_epoch']}: {links}.")
    lines.extend(["", "[Frozen selections, commands and hashes](manifest.json), [metrics CSV](metrics.csv), [metrics JSON](metrics.json). "
                  "Per-model logs are saved alongside this file.", ""])
    (ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "report_section.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("Wrote step 1 and step 2 tables, per-model artifact links, metrics CSV/JSON and a report section.")


if __name__ == "__main__":
    main()
