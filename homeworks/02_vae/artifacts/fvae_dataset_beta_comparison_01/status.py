"""Read-only progress for the requested f-VAE dataset/beta matrix."""
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
root = next(path for path in folder.parents if (path / "pyproject.toml").exists())
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
active = next((entry for entry in manifest["experiments"] if entry["status"] == "running"), None)
result = {"status": manifest["status"],
          "completed_new": sum(entry["status"] == "complete" for entry in manifest["experiments"]),
          "reused": sum(entry["reused"] for entry in manifest["experiments"]), "active": active["label"] if active else None}
if active:
    try:
        history = json.loads((root / active["output"] / "history.json").read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["validation"]["negative_elbo"])
        result.update(epoch=history[-1]["epoch"], validation=history[-1]["validation"],
                      best_epoch=best["epoch"], best_negative_elbo=best["validation"]["negative_elbo"])
    except (FileNotFoundError, json.JSONDecodeError):
        result["epoch"] = 0
print(json.dumps(result))
