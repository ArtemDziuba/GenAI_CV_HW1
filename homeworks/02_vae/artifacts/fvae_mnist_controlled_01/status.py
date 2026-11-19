"""Read progress for this bounded experiment batch."""
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
root = next(path for path in folder.parents if (path / "pyproject.toml").exists())
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
entry = manifest["runs"][-1]
result = {"status": manifest["status"], "run": entry["label"],
          "completed_runs": [run["label"] for run in manifest["runs"] if run["status"] == "complete"]}
try:
    history = json.loads((root / entry["output"] / "history.json").read_text(encoding="utf-8"))
    best = min(history, key=lambda row: row["validation"]["negative_elbo"])
    result.update(epoch=history[-1]["epoch"], latest_validation=history[-1]["validation"],
                  best_epoch=best["epoch"], best_negative_elbo=best["validation"]["negative_elbo"])
except (FileNotFoundError, json.JSONDecodeError):
    result["epoch"] = 0
print(json.dumps(result))
