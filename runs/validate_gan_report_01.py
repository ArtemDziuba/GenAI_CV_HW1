"""Read-only checks for report structure, links, measured tables and sources."""

import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "homeworks/03_gan/report.md"
ARTIFACTS = REPORT.parent / "artifacts"
text = REPORT.read_text(encoding="utf-8")
plain = text.replace("**", "")
expected_sections = ["Implementation and experimental setup", "Results and required artifacts",
                     "What worked well", "What did not work as expected",
                     "Possible explanations for limitations", "Potential improvements"]
assert re.findall(r"^## (.+)$", text, re.M) == expected_sections
assert not re.search(r"TODO|scaffold only", text, re.I)

links = [(path, target) for path in (REPORT, ARTIFACTS / "index.md")
         for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
         if not target.startswith(("https://", "http://", "#"))]
missing = [(str(path), target) for path, target in links if not (path.parent / target.split("#")[0]).is_file()]
assert not missing, missing

readme = (ROOT / "README.md").read_text(encoding="utf-8")
assignment = readme[readme.index("## GAN - 10pt. (+2pt.)"):].strip().splitlines()[1:]
quote = "\n".join(line[2:] if line.startswith("> ") else "" for line in text.splitlines() if line.startswith(">"))
assert all(line.strip() in quote for line in assignment if line.strip())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


before = json.loads((ROOT / "runs/gan_report_source_hashes_01.json").read_text())
assert all(digest(ROOT / name) == value for name, value in before.items() if (ROOT / name).resolve() != REPORT)
provenance = json.loads((ARTIFACTS / "report_provenance.json").read_text())
assert all(digest(ROOT / name) == value for name, value in provenance["source_hashes"].items())
representatives = json.loads((ARTIFACTS / "report_metrics.json").read_text())
assert len(representatives) == 13
assert all(f"{row['kid']:.4f}" in text for row in representatives)


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


checked_rows = 0
for row in csv_rows(ARTIFACTS / "mnist_width_comparison_01.csv"):
    if row["epoch_budget"] == "30":
        expected = f"| {row['gc']}/{row['dc']} | {row['selected_epoch']} | {float(row['kid']):.4f} |"
        assert expected in plain, expected
        checked_rows += 1
sweep = csv_rows(ARTIFACTS / "gan_mnist_sweep_01/metrics.csv")
for row in sweep:
    if row["epochs"] == "60" and row["gc"] == "256" and row["dc"] == "64" and row["batch"] == "64":
        expected = f"| {float(row['g_lr']):.4f} | {float(row['d_lr']):.4f} | {row['best_epoch']} | {float(row['kid']):.4f} |"
        assert expected in plain, expected
        checked_rows += 1
    if row["role"].startswith("top_three_rank"):
        expected = f"| {float(row['g_lr']):.4f}/{float(row['d_lr']):.4f} | {row['best_epoch']} | {float(row['kid']):.4f} |"
        assert expected in plain, expected
        checked_rows += 1
for row in csv_rows(ARTIFACTS / "fashion_classifier_upgrade_01/metrics.csv"):
    expected = (f"| {row['gc']}/{row['dc']} | {row['stored_checkpoint_epoch']} | {float(row['v1_kid_separate_scale']):.4f} | "
                f"{float(row['v2_kid']):.4f} | {100*float(row['v2_confident_fraction']):.2f}% |")
    assert expected in plain, expected
    checked_rows += 1
variants = json.loads((ARTIFACTS / "cgan_variants_01/metrics.json").read_text())
names = {"baseline": "Global cGAN baseline, first 30 epochs", "patch": "PatchGAN",
         "fm10": "Global cGAN + feature matching", "patch_fm10": "PatchGAN + feature matching"}
for key, name in names.items():
    digits = next(row for row in variants if row["dataset"] == "mnist" and row["variant"] == key)
    fashion = next(row for row in variants if row["dataset"] == "fashion_mnist" and row["variant"] == key)
    expected = f"| {name} | {digits['best_epoch']} | {digits['kid']:.4f} | {fashion['best_epoch']} | {fashion['kid']:.4f} |"
    assert expected in plain, expected
    checked_rows += 1

print(f"PASS: six template sections, full assignment quote, {len(links)} local links/images, "
      f"{checked_rows} exact experimental table rows and 13 representative metrics.")
print("PASS: README, previous reports, GAN sources, original run records and saved checkpoint hashes unchanged.")
print(f"Report: {len(text.split())} words, {len(text.splitlines())} lines. No training or inference performed.")
