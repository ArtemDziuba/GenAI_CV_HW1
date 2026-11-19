"""AE analysis uses saved best-checkpoint metrics without running models."""

from copy import deepcopy
import json

from vae.compare import best_compression_runs, compression_groups, export_compression


def ae_row(name, latent, mse, **overrides):
    return {
        "run": name, "model": "ae", "dataset": "mnist", "latent_dim": latent,
        "hidden_channels": 16, "hidden_dim": 64, "conv_depth": 1,
        "ae_loss": "mse", "batch_size": 64, "learning_rate": 0.001,
        "epochs": 30, "weight_decay": 0., "grad_clip": 0.,
        "validation_fraction": 0.1, "train_size": 54000, "validation_size": 6000,
        "seed": 42, "deterministic": False, "best_epoch": 23,
        "validation_grayscale_mse": mse, **overrides,
    }


def test_latent_comparison_separates_confounding_controls():
    rows = [ae_row("z16", 16, .01), ae_row("z32", 32, .008)]
    for override in ({"hidden_channels": 32}, {"ae_loss": "l1"}, {"seed": 7},
                     {"batch_size": 256}, {"train_size": 1024}, {"dataset": "fashion_mnist"},
                     {"learning_rate": .01}, {"deterministic": True}):
        rows.append(ae_row(str(override), 32, .005, **override))
    previous = deepcopy(rows)
    groups = compression_groups(rows)
    pairs = [group for group in groups if len(group) > 1]
    assert len(groups) == 9
    assert [[row["run"] for row in pair] for pair in pairs] == [["z16", "z32"]]
    assert rows == previous


def test_best_results_are_selected_per_dataset_and_latent_not_globally():
    rows = [ae_row("small16", 16, .01), ae_row("wide16", 16, .006, hidden_channels=32),
            ae_row("small32", 32, .008),
            ae_row("fashion16", 16, .02, dataset="fashion_mnist"),
            ae_row("vae", 16, .0001, model="vae")]
    assert {row["run"] for row in best_compression_runs(rows)} == {"wide16", "small32", "fashion16"}


def test_export_preserves_single_points_and_documents_uncontrolled_best(tmp_path):
    rows = [ae_row("missing_weights_z16", 16, .01), ae_row("missing_weights_z32", 32, .008),
            ae_row("single_wide", 64, .003, hidden_channels=32),
            ae_row("single_fashion", 32, .02, dataset="fashion_mnist")]
    previous = deepcopy(rows)
    # No checkpoints or datasets exist: analysis must need only these saved metrics.
    export_compression(rows, tmp_path)
    groups = json.loads((tmp_path / "compression_groups.json").read_text())
    assert len(groups) == 3
    assert sorted(len(group["latent_sizes"]) for group in groups) == [1, 1, 2]
    assert len(list((tmp_path / "by_configuration").glob("*.png"))) == 3
    for name in ("compression.png", "compression_best.png", "compression_mnist.png", "compression_fashion_mnist.png"):
        assert (tmp_path / name).stat().st_size > 0
    report = (tmp_path / "compression.md").read_text(encoding="utf-8")
    assert "not the isolated effect of Z" in report
    assert "single measured latent size; more runs needed for a curve" in report
    assert "missing_weights_z32" in report
    assert rows == previous
