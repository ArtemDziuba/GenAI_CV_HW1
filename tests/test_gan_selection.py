"""Loss minima and visual preference remain distinct; missing weights are honest."""

import hashlib
import json

import pytest
import torch

from gan.selection import checkpoint_scores, loss_minima, visual_selection
import gan.select as select


def history():
    return [{"epoch": 1, "train": {"g_loss": 2., "d_loss": .4}, "validation": {"g_loss": 1., "d_loss": .6}},
            {"epoch": 2, "train": {"g_loss": 1., "d_loss": .5}, "validation": {"g_loss": .8, "d_loss": .5}},
            {"epoch": 3, "train": {"g_loss": 1., "d_loss": .6}, "validation": {"g_loss": 1.2, "d_loss": .4}}]


def test_minima_keep_earliest_ties_and_mark_unsaved_epochs(tmp_path):
    torch.save({"epoch": 1}, tmp_path / "epoch_001.pt")
    torch.save({"epoch": 3}, tmp_path / "last.pt")
    records = loss_minima(history(), tmp_path)["minima"]
    assert records["train"]["g_loss"] == {"epoch": 2, "value": 1., "checkpoint": None, "checkpoint_available": False}
    assert records["validation"]["g_loss"]["epoch"] == 2
    assert records["train"]["d_loss"]["checkpoint"] == "epoch_001.pt"
    assert records["validation"]["d_loss"]["checkpoint"] == "last.pt"


def test_visual_choice_need_not_minimize_loss_and_preserves_source_files(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(history()))
    torch.save({"epoch": 1}, tmp_path / "epoch_001.pt")
    before = hashlib.sha256((tmp_path / "epoch_001.pt").read_bytes()).hexdigest()
    select.main(["--run", str(tmp_path), "--epoch", "1", "--reason", "Preferred stroke shapes"])
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["best_epoch"] == 1 and selection["selection_method"] == "manual_visual"
    assert selection["checkpoint_sha256"] == before
    assert selection["validation"]["g_loss"] != min(row["validation"]["g_loss"] for row in history())
    assert hashlib.sha256((tmp_path / "epoch_001.pt").read_bytes()).hexdigest() == before
    assert json.loads((tmp_path / "history.json").read_text()) == history()
    assert (tmp_path / "loss_minima.json").is_file()
    with pytest.raises(FileExistsError):
        select.main(["--run", str(tmp_path), "--epoch", "1", "--reason", "Another choice"])


def test_unsaved_epoch_cannot_be_selected_or_recovered_from_last(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(history()))
    torch.save({"epoch": 3}, tmp_path / "last.pt")
    with pytest.raises(FileNotFoundError, match="history alone cannot recover"):
        visual_selection(tmp_path, 2, "Lower diagnostic loss")
    assert not (tmp_path / "selection.json").exists()


def test_minima_refresh_does_not_create_visual_selection(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(history()))
    select.main(["--run", str(tmp_path)])
    assert (tmp_path / "loss_minima.json").is_file()
    assert not (tmp_path / "selection.json").exists()


def test_checkpoint_metadata_must_match_visual_epoch(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(history()))
    torch.save({"epoch": 3}, tmp_path / "epoch_001.pt")
    with pytest.raises(ValueError, match="does not match"):
        visual_selection(tmp_path, 1, "Preferred preview")


def test_visual_selection_and_minima_resolve_retained_best_checkpoint(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps(history()))
    torch.save({"epoch": 2}, tmp_path / "best_g.pt")
    (tmp_path / "best_checkpoints.json").write_text(json.dumps({"g": {"epoch": 2, "checkpoint": "best_g.pt"}}))
    selection = visual_selection(tmp_path, 2, "Preferred retained checkpoint")
    assert selection["checkpoint"] == "best_g.pt"
    assert loss_minima(history(), tmp_path)["minima"]["validation"]["g_loss"]["checkpoint"] == "best_g.pt"


def test_checkpoint_scores_use_total_losses_and_equal_weight_sum():
    scores = checkpoint_scores({"g_loss": 2., "d_loss": .5, "g_adversarial_loss": 1.})
    assert scores == {"g": 2., "d": .5, "gd": 2.5}
    with pytest.raises(ValueError, match="nonfinite"):
        checkpoint_scores({"g_loss": float("nan"), "d_loss": .5})
