"""Resume paired networks/Adam states, preserve the starting best, and count epochs."""

import hashlib
import json

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from shared.data import DataLoaders
import gan.train as train
import gan.artifacts as artifacts


pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


@pytest.fixture
def resume_loaders(monkeypatch, synthetic_dataset):
    dataset = Subset(synthetic_dataset, range(6))
    def make(*args, **kwargs):
        training = DataLoader(dataset, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(42))
        validation = DataLoader(dataset, batch_size=3, shuffle=False, generator=torch.Generator().manual_seed(43))
        return DataLoaders(training, validation, validation)
    monkeypatch.setattr(train, "make_dataloaders", make)
    return make


def command(bundle, output, epochs):
    return ["--evaluation-set", str(bundle), "--output", str(output), "--epochs", str(epochs),
            "--device", "cpu", "--generator-channels", "2", "--discriminator-channels", "2",
            "--latent-dim", "4", "--batch-size", "3", "--sample-count", "2", "--train-limit", "6"]


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def controlled_scores(monkeypatch, scores):
    original = train.GeneratorEvaluator.evaluate
    # Unwrap a previous score wrapper when replacing it in the same test.
    original = getattr(original, "actual_evaluate", original)
    values = iter(scores)
    def evaluate(evaluator, generator):
        metrics = original(evaluator, generator)
        metrics["kid"] = next(values)
        return metrics
    evaluate.actual_evaluate = original
    monkeypatch.setattr(train.GeneratorEvaluator, "evaluate", evaluate)


def test_resume_matches_uninterrupted_cpu_training_and_restores_both_adams(tmp_path, monkeypatch, resume_loaders, gan_evaluation_bundle):
    full, parent, continued = (tmp_path / name for name in ("full", "parent", "continued"))
    controlled_scores(monkeypatch, [4., 3., 2., 1.])
    train.main(command(gan_evaluation_bundle, full, 4))
    controlled_scores(monkeypatch, [4., 3.])
    train.main(command(gan_evaluation_bundle, parent, 2))
    source = parent / "best_kid.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    figures = []
    original_plot = artifacts.plot_metrics
    def capture_plot(*args, **kwargs):
        figure = original_plot(*args, **kwargs)
        figures.append(figure)
        return figure
    monkeypatch.setattr(artifacts, "plot_metrics", capture_plot)
    controlled_scores(monkeypatch, [3., 2., 1.])  # Restored baseline, then epochs 3 and 4.
    # Omitted architecture/optimizer flags inherit the saved settings.
    train.main(["--resume", str(source), "--evaluation-set", str(gan_evaluation_bundle),
                "--epochs", "4", "--device", "cpu", "--output", str(continued)])
    expected = torch.load(full / "best_kid.pt", weights_only=True)
    actual = torch.load(continued / "best_kid.pt", weights_only=True)
    for key in ("generator_state", "discriminator_state", "generator_optimizer", "discriminator_optimizer", "training_state"):
        assert_tree_equal(expected[key], actual[key])
    assert actual["epoch"] == 4
    assert all(actual["config"]["resumption"]["restored_rngs"][key] for key in ("torch_rng", "train_loader_rng", "validation_loader_rng"))
    history = json.loads((continued / "history.json").read_text())
    assert [row["epoch"] for row in history] == [3, 4]
    summary = json.loads((continued / "summary.json").read_text())
    assert summary["g_updates"] == 8 and summary["session_g_updates"] == 4
    assert summary["epochs_trained_this_run"] == 2 and summary["first_training_epoch"] == 3
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert figures
    for figure in figures:
        for line in figure.axes[0].lines:
            assert list(line.get_xdata()) == list(range(3, 3 + len(line.get_xdata())))


def test_legacy_resume_lr_override_and_retention_when_new_epochs_worsen(tmp_path, monkeypatch, resume_loaders, gan_evaluation_bundle):
    parent, continued = tmp_path / "parent", tmp_path / "continued"
    controlled_scores(monkeypatch, [1.])
    train.main(command(gan_evaluation_bundle, parent, 1))
    source = parent / "best_kid.pt"
    legacy = torch.load(source, weights_only=True)
    legacy.pop("training_state")
    torch.save(legacy, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original_epoch = train.run_epoch
    checked = []
    def inspect(generator, discriminator, loader, device, **kwargs):
        if kwargs.get("generator_optimizer") is not None:
            assert_tree_equal(generator.state_dict(), legacy["generator_state"])
            assert_tree_equal(discriminator.state_dict(), legacy["discriminator_state"])
            g = kwargs["generator_optimizer"].state_dict()
            d = kwargs["discriminator_optimizer"].state_dict()
            assert_tree_equal(g["state"], legacy["generator_optimizer"]["state"])
            assert_tree_equal(d["state"], legacy["discriminator_optimizer"]["state"])
            assert g["param_groups"][0]["lr"] == .0001
            assert d["param_groups"][0]["lr"] == .0002
            checked.append(True)
        return original_epoch(generator, discriminator, loader, device, **kwargs)
    monkeypatch.setattr(train, "run_epoch", inspect)
    controlled_scores(monkeypatch, [1., 2.])
    train.main(["--resume", str(source), "--evaluation-set", str(gan_evaluation_bundle), "--output", str(continued),
                "--device", "cpu", "--epochs", "2", "--generator-lr", ".0001"])
    assert checked == [True]
    saved = torch.load(continued / "best_kid.pt", weights_only=True)
    assert saved["epoch"] == 1 and saved["selection"]["from_resume"]
    assert_tree_equal(saved["generator_state"], legacy["generator_state"])
    assert_tree_equal(saved["discriminator_state"], legacy["discriminator_state"])
    assert not any(saved["config"]["resumption"]["restored_rngs"].values())
    summary = json.loads((continued / "summary.json").read_text())
    assert summary["last_epoch"] == 2 and summary["best_epoch"] == 1 and summary["train"] is None
    assert (continued / "samples_epoch_002.png").is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("changes", [["--generator-channels", "3"], ["--dataset", "fashion_mnist"],
                                    ["--batch-size", "4"], ["--feature-matching-weight", "1"],
                                    ["--epochs", "1"], ["--evaluation-seed", "45"]])
def test_resume_rejects_incompatible_experiments(tmp_path, monkeypatch, resume_loaders, gan_evaluation_bundle, changes):
    parent, rejected = tmp_path / "parent", tmp_path / "rejected"
    controlled_scores(monkeypatch, [1.])
    train.main(command(gan_evaluation_bundle, parent, 1))
    with pytest.raises(SystemExit):
        train.main(["--resume", str(parent / "best_kid.pt"), "--evaluation-set", str(gan_evaluation_bundle),
                    "--epochs", "2", "--device", "cpu", "--output", str(rejected), *changes])
    assert not rejected.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_resume_restores_optimizer_tensors_and_advances_steps(tmp_path, monkeypatch, resume_loaders, gan_evaluation_bundle):
    parent, continued = tmp_path / "parent", tmp_path / "continued"
    controlled_scores(monkeypatch, [2.])
    train.main([*command(gan_evaluation_bundle, parent, 1), "--device", "cuda"])
    original_epoch = train.run_epoch
    inspected = []
    def inspect(generator, discriminator, loader, device, **kwargs):
        if kwargs.get("generator_optimizer") is not None:
            for name in ("generator_optimizer", "discriminator_optimizer"):
                for state in kwargs[name].state.values():
                    assert state["exp_avg"].is_cuda and state["exp_avg_sq"].is_cuda
                    assert state["step"].item() == 2
            inspected.append(True)
        return original_epoch(generator, discriminator, loader, device, **kwargs)
    monkeypatch.setattr(train, "run_epoch", inspect)
    controlled_scores(monkeypatch, [2., 1.])
    train.main(["--resume", str(parent / "best_kid.pt"), "--evaluation-set", str(gan_evaluation_bundle),
                "--output", str(continued), "--epochs", "2", "--device", "cuda"])
    assert inspected == [True]
    saved = torch.load(continued / "best_kid.pt", map_location="cpu", weights_only=True)
    assert saved["config"]["resumption"]["restored_rngs"]["cuda_rng"]
    assert all(state["step"].item() == 4 for state in saved["generator_optimizer"]["state"].values())
