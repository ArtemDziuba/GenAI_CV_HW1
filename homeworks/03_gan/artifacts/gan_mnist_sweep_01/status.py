"""Read sweep progress without loading models or touching training outputs."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
print("Sweep:", plan["status"])
for trial in plan["trials"]:
    if trial["status"] == "complete":
        print(f"{trial['id']}: complete; KID {trial['kid']:.4f} at epoch {trial['best_epoch']}")
    elif trial["status"] == "running":
        progress = trial.get("progress", {})
        print(f"{trial['id']}: epoch {progress.get('epoch', 0)}/{trial['epochs']}; "
              f"best KID {progress.get('best_kid', 'pending')} at {progress.get('best_epoch', '-')}")
    else:
        print(trial["id"], trial["status"])
for decision in plan["decisions"]:
    print("Decision:", decision)
