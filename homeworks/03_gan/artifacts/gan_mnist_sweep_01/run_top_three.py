"""Confirm the three best distinct 60-epoch configurations for 100 epochs.

Authorized after the final two rate trials. Completed matching 100-epoch runs
are reused; no test results or interrupted-run scores enter the ranking.
"""

from datetime import UTC, datetime

from run_batch import MANIFEST, config_key, read, run_trial, verify_frozen, write
from export_analysis import main as export_analysis


def main() -> None:
    plan = read(MANIFEST)
    if plan["status"] != "learning_rate_grid_complete_confirmation_deferred":
        raise RuntimeError("Finish the two remaining rate trials before selecting the top three")
    verify_frozen(plan)
    reference = {"id": "existing_gc128_dc64_b64_e60", "settings": plan["base"], "epochs": 60,
                 "kid": plan["reference"]["summary"]["evaluation"]["kid"], "run": plan["reference"]["run"]}
    candidates = [reference, *(trial for trial in plan["trials"]
                               if trial["status"] == "complete" and trial["epochs"] == 60)]
    if len({config_key(trial["settings"], 60) for trial in candidates}) != len(candidates):
        raise RuntimeError("Duplicate completed 60-epoch configurations need explicit ranking policy")
    ranked = sorted(candidates, key=lambda trial: (trial["kid"], trial["id"]))
    selected = ranked[:3]
    if len(selected) != 3:
        raise RuntimeError("Need three completed distinct configurations")
    plan["decisions"].append({
        "stage": "top_three_for_100", "selection": "Lowest validation KID among completed distinct 60-epoch configurations",
        "candidates": [{key: trial[key] for key in ("id", "settings", "kid")} for trial in ranked],
        "selected": [trial["id"] for trial in selected],
        "matching_completed_100_policy": "Reuse, do not train twice",
        "authorized_utc": datetime.now(UTC).isoformat(),
    })
    plan["status"] = "running_top_three_100"
    plan["pending"] = [{"stage": "top_three_100", "settings": trial["settings"], "epochs": 100} for trial in selected]
    write(plan)
    for rank, trial in enumerate(selected, 1):
        print(f"SELECT rank {rank}: {trial['id']}, 60-epoch KID {trial['kid']:.6f}", flush=True)
    confirmations = []
    for rank, trial in enumerate(selected, 1):
        confirmed = run_trial(plan, f"top_three_rank{rank}_100", trial["settings"], 100)
        if confirmed["status"] != "complete":
            raise RuntimeError("A top-three confirmation failed; inspect its recorded log")
        confirmations.append({"rank_at_60": rank, "source_60": trial["id"], "confirmation_100": confirmed["id"],
                              "settings": trial["settings"], "kid_60": trial["kid"], "kid_100": confirmed["kid"],
                              "best_epoch_100": confirmed["best_epoch"]})
        plan["top_three_confirmations"] = confirmations
        plan["pending"] = [{"stage": "top_three_100", "settings": remaining["settings"], "epochs": 100}
                           for remaining in selected[rank:]]
        write(plan)
    verify_frozen(plan)
    plan.update(status="complete_requested_experiments", finished_utc=datetime.now(UTC).isoformat())
    write(plan)
    export_analysis()
    print("DONE: final two rate trials and all three requested 100-epoch confirmations completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        plan = read(MANIFEST)
        plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        write(plan)
        raise
