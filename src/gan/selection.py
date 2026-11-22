"""Record loss minima separately from a student's visual checkpoint selection."""

import hashlib
import json
import math
from pathlib import Path

import torch


def checkpoint_scores(validation: dict) -> dict[str, float]:
    """Equal-weight G+D uses the same logged total G and D objectives."""
    scores = {"g": validation["g_loss"], "d": validation["d_loss"],
              "gd": validation["g_loss"] + validation["d_loss"]}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("nonfinite checkpoint selection score")
    return scores


def _checkpoint_for_epoch(run: Path, epoch: int, last_epoch: int) -> Path | None:
    """Find a retained full G/D snapshot, including legacy run layouts."""
    path = run / f"epoch_{epoch:03d}.pt"
    if path.is_file():
        return path
    records_path = run / "best_checkpoints.json"
    if records_path.is_file():
        records = json.loads(records_path.read_text(encoding="utf-8"))
        for record in records.values():
            if record["epoch"] == epoch and (run / record["checkpoint"]).is_file():
                return run / record["checkpoint"]
    if epoch == last_epoch and (run / "last.pt").is_file():
        return run / "last.pt"
    return None


def loss_minima(history: list[dict], run: Path) -> dict:
    """Earliest minimum for each G/D loss; unavailable old weights stay unavailable.

    A smaller D loss means a better discriminator objective, not better generated
    images. These records are diagnostics, not an automatic image-quality ranking.
    """
    if not history:
        raise ValueError("cannot summarize empty history")
    records = {}
    for split in ("train", "validation"):
        records[split] = {}
        for metric in ("g_loss", "d_loss"):
            if not all(math.isfinite(row[split][metric]) for row in history):
                raise ValueError("nonfinite recorded loss")
            row = min(history, key=lambda entry: entry[split][metric])
            path = _checkpoint_for_epoch(run, row["epoch"], history[-1]["epoch"])
            checkpoint = path.name if path is not None else None
            records[split][metric] = {"epoch": row["epoch"], "value": row[split][metric],
                                      "checkpoint": checkpoint, "checkpoint_available": checkpoint is not None}
    return {"minima": records, "note": "Loss minima are diagnostic records, not best generation epochs."}


def visual_selection(run: Path, epoch: int, reason: str) -> dict:
    """Reference an existing checkpoint without copying or overwriting its weights."""
    if epoch < 1 or not reason.strip():
        raise ValueError("a positive epoch and a nonempty visual-selection reason are required")
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    row = next((item for item in history if item["epoch"] == epoch), None)
    if row is None:
        raise ValueError(f"epoch {epoch} is absent from history")
    checkpoint = _checkpoint_for_epoch(run, epoch, history[-1]["epoch"])
    if checkpoint is None:
        raise FileNotFoundError(f"Weights for epoch {epoch} are not retained; history alone cannot recover them")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["epoch"] != epoch:
        raise ValueError("checkpoint epoch does not match requested selection")
    grid = f"samples_epoch_{epoch:03d}.png"
    return {"best_epoch": epoch, "selection_method": "manual_visual", "reason": reason.strip(),
            "checkpoint": checkpoint.name, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "sample_grid": grid if (run / grid).is_file() else None,
            "train": row["train"], "validation": row["validation"],
            "note": "Visual preference recorded explicitly; this is not a claim of minimum GAN loss or test-set selection."}
