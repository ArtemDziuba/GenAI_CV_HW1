"""Record a visual epoch choice, or refresh loss minima, without training.

Example: python -m gan.select --run checkpoints/gan/my_run --epoch 25
    --reason "Preferred fixed-noise generations over epoch 30"
"""

import argparse
import json
from pathlib import Path

from shared.experiment import write_json
from .selection import loss_minima, visual_selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, help="Visually selected epoch; omit to refresh loss records only")
    parser.add_argument("--reason", help="Explain the visual preference")
    parser.add_argument("--output", type=Path, help="New selection JSON; defaults to RUN/selection.json")
    args = parser.parse_args(argv)
    if args.epoch is None and (args.reason is not None or args.output is not None):
        parser.error("--reason and --output require --epoch")
    if args.epoch is not None and (args.epoch < 1 or not args.reason or not args.reason.strip()):
        parser.error("--epoch must be positive and --reason must describe the visual preference")
    if args.epoch is not None:
        selection = visual_selection(args.run, args.epoch, args.reason)
        output = args.output or args.run / "selection.json"
        # Preserve previous visual choices unless a different output is requested.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(selection, indent=2, allow_nan=False) + "\n")
        print(f"Recorded visually selected epoch {args.epoch}: {output}")
    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))
    minima = loss_minima(history, args.run)
    write_json(args.run / "loss_minima.json", minima)
    for split, metrics in minima["minima"].items():
        for metric, row in metrics.items():
            print(f"Lowest {split} {metric}: epoch {row['epoch']}, {row['value']:.6f}, checkpoint={row['checkpoint']}")


if __name__ == "__main__":
    main()
