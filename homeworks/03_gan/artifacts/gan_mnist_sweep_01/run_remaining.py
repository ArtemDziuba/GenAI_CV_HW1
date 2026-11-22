"""Run only the last two user-authorized 60-epoch learning-rate trials."""

from datetime import UTC, datetime

from run_batch import MANIFEST, choose, config_key, read, run_trial, verify_frozen, write
from export_analysis import main as export_analysis


def main() -> None:
    plan = read(MANIFEST)
    verify_frozen(plan)
    base = next(item["settings"] for item in plan["decisions"] if item["stage"] == "learning_rate_base_100")
    plan.setdefault("resumptions", []).append({
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "Only final two 60-epoch rate trials; no 100-epoch training",
        "previous_status": plan["status"],
        "interrupted_run_policy": "Preserve the partial trial; restart its configuration in a fresh directory",
    })
    plan["status"] = "running_remaining_rate_trials"
    write(plan)
    first = run_trial(plan, "lr_g400u_d200u_60_restart01", {**base, "g_lr": .0004, "d_lr": .0002},
                      60, restart_suffix="_restart01")
    second = run_trial(plan, "lr_g400u_d400u_60", {**base, "g_lr": .0004, "d_lr": .0004}, 60)
    if any(trial["status"] != "complete" for trial in (first, second)):
        raise RuntimeError("A remaining trial failed; inspect its recorded log")
    rates = (.0001, .0002, .0004)
    candidates = []
    for g_lr in rates:
        for d_lr in rates:
            settings = {**base, "g_lr": g_lr, "d_lr": d_lr}
            matches = [trial for trial in plan["trials"] if trial["status"] == "complete"
                       and config_key(trial["settings"], trial["epochs"]) == config_key(settings, 60)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one complete grid cell for {g_lr}/{d_lr}")
            candidates.append(matches[0])
    choose(plan, "learning_rates_60", candidates)
    verify_frozen(plan)
    plan.update(status="learning_rate_grid_complete_confirmation_deferred",
                rate_grid_finished_utc=datetime.now(UTC).isoformat(),
                pending=[{"stage": "learning_rate_winner_100", "status": "deferred; latest authorization covered only two 60-epoch trials"}])
    write(plan)
    export_analysis()
    print("DONE: the final two trials and full grid comparison are complete. No further training will start.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        plan = read(MANIFEST)
        plan.update(status="stopped_error", error=f"{type(error).__name__}: {error}")
        write(plan)
        raise
