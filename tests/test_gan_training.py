"""GAN CLI artifacts and checkpoint round trips using offline synthetic data."""

import csv
import json

import pytest
import torch
from torch.utils.data import DataLoader

from shared.data import DataLoaders
import gan.sample as sample
import gan.train as train
import gan.evaluate as evaluate


pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


@pytest.mark.parametrize("options", [[], ["--conditional"], ["--discriminator", "patch"],
                                     ["--feature-matching-weight", "10"],
                                     ["--conditional", "--discriminator", "patch", "--feature-matching-weight", "10"]])
def test_cli_artifacts_and_checkpoint_round_trip(monkeypatch, tmp_path, synthetic_dataset, gan_evaluation_bundle, options):
    loader = DataLoader(torch.utils.data.Subset(synthetic_dataset, range(5)), batch_size=3)
    monkeypatch.setattr(train, "make_dataloaders", lambda *args, **kwargs: DataLoaders(loader, loader, loader))
    output = tmp_path / "run"
    command = ["--device", "cpu", "--epochs", "2", "--generator-channels", "2", "--discriminator-channels", "2",
               "--latent-dim", "4", "--embedding-dim", "3", "--sample-count", "4", "--samples-per-class", "2",
               "--sample-every", "2", "--discriminator-steps", "2", "--output", str(output),
               "--evaluation-set", str(gan_evaluation_bundle), "--train-limit", "5", *options]
    train.main(command)
    for name in ("config.json", "summary.json", "history.json", "best_checkpoints.json", "best_kid.pt",
                 "preview_inputs.pt", "steps.csv", "losses.png", "generator_terms.png", "discriminator_probabilities.png",
                 "discriminator_accuracy.png", "samples_epoch_000.png", "samples_epoch_001.png", "samples_last.png", "real_validation.png",
                 "samples_best_kid.png", "kid.png", "class_coverage.png", "class_entropy.png", "classifier_confidence.png",
                 "evaluation_history.json", "evaluation.csv", "evaluation_inputs.pt", "evaluation_protocol.json"):
        assert (output / name).stat().st_size > 0
    assert not (output / "best.pt").exists()
    assert {path.name for path in output.glob("*.pt")} == {"best_kid.pt", "preview_inputs.pt", "evaluation_inputs.pt"}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["last_epoch"] == 2 and summary["selection_metric"] == "kid"
    assert set(summary["best_checkpoints"]) == {"kid"}
    assert summary["evaluation"]["samples"] == summary["evaluation"]["real_samples"] == 20
    assert not (output / "loss_minima.json").exists()
    assert summary["g_updates"] == 4 and summary["d_updates"] == 8
    rows = list(csv.DictReader((output / "steps.csv").open()))
    assert len(rows) == 4 and int(rows[-1]["global_d_step"]) == 8
    assert int(rows[-1]["batch_images"]) == 2
    history = json.loads((output / "history.json").read_text())
    assert len(history) == 2 and history[0]["train"]["images"] == 5
    assert history[0]["validation"]["g_updates"] == 0
    generator, discriminator, checkpoint = train.load_checkpoint(output / "best_kid.pt", torch.device("cpu"))
    assert checkpoint["epoch"] == summary["best_epoch"] and not generator.training and not discriminator.training
    assert checkpoint["generator_optimizer"] and checkpoint["discriminator_optimizer"]
    fixed = torch.load(output / "preview_inputs.pt", weights_only=True)
    with torch.no_grad():
        first = generator(fixed["noise"], fixed["labels"])
        restored, _, _ = train.load_checkpoint(output / "best_kid.pt", torch.device("cpu"))
        torch.testing.assert_close(first, restored(fixed["noise"], fixed["labels"]))
    with pytest.raises(FileExistsError):
        train.main(command)
    destination = tmp_path / "new_samples"
    sample.main(["--checkpoint", str(output / "best_kid.pt"), "--output", str(destination), "--device", "cpu",
                 "--sample-count", "4", "--samples-per-class", "2"])
    metadata = json.loads((destination / "samples.json").read_text())
    assert metadata["count"] == (20 if "--conditional" in options else 4)
    assert (destination / "samples.png").stat().st_size > 0
    evaluated = tmp_path / "evaluated"
    evaluate.main(["--checkpoint", str(output / "best_kid.pt"), "--evaluation-set", str(gan_evaluation_bundle),
                   "--device", "cpu", "--output", str(evaluated), "--sample-count", "4", "--samples-per-class", "2"])
    result = json.loads((evaluated / "evaluation.json").read_text())
    assert result["metrics"]["kid"] == pytest.approx(summary["evaluation"]["kid"], abs=1e-12)


@pytest.mark.parametrize("options", [["--generator-lr", "nan"], ["--adam-beta1", "1"],
                                     ["--discriminator-steps", "0"], ["--feature-matching-weight", "-1"],
                                     ["--sample-every", "0"], ["--seed", "-1"], ["--validation-fraction", "1"]])
def test_invalid_training_options(options):
    with pytest.raises(SystemExit):
        train.main(["--evaluation-set", "unused", *options])


def test_kid_selects_checkpoint_and_matching_grid_independent_of_gan_losses(monkeypatch, tmp_path, synthetic_dataset, gan_evaluation_bundle):
    loader = DataLoader(torch.utils.data.Subset(synthetic_dataset, range(2)), batch_size=2)
    monkeypatch.setattr(train, "make_dataloaders", lambda *args, **kwargs: DataLoaders(loader, loader, loader))
    validation = iter([(.5, 2.), (1., .1), (.6, .4), (.6, .4)])
    scores = iter([.4, -.2, -.2, .3])
    original_evaluate = train.GeneratorEvaluator.evaluate
    def fake_evaluate(evaluator, generator):
        result = original_evaluate(evaluator, generator)
        result['kid'] = next(scores)
        return result
    monkeypatch.setattr(train.GeneratorEvaluator, "evaluate", fake_evaluate)
    epoch = 0
    preview_calls = {}
    original_save_samples = train.save_samples
    def capture_preview(generator, noise, labels, path, **kwargs):
        preview_calls.setdefault(path.name, []).append((next(generator.parameters()).detach().clone(), noise.clone()))
        return original_save_samples(generator, noise, labels, path, **kwargs)
    monkeypatch.setattr(train, "save_samples", capture_preview)
    def fake_epoch(generator, discriminator, loader, device, *, generator_optimizer=None, on_step=None, **kwargs):
        nonlocal epoch
        if generator_optimizer is not None:
            epoch += 1
            with torch.no_grad():
                next(generator.parameters()).fill_(epoch)
                next(discriminator.parameters()).fill_(-epoch)
            g, d = 5., 5.  # Training minima must not select validation checkpoints.
        else:
            g, d = next(validation)
        return {"g_loss": g, "d_loss": d, "g_adversarial_loss": g, "g_feature_matching_loss": 0.,
                "d_real_probability": .5, "d_fake_probability": .5, "g_fake_probability": .5, "d_accuracy": .5}
    monkeypatch.setattr(train, "run_epoch", fake_epoch)
    output = tmp_path / "kid_model"
    train.main(["--device", "cpu", "--epochs", "4", "--sample-every", "5", "--output", str(output),
                "--generator-channels", "2", "--discriminator-channels", "2", "--latent-dim", "4", "--sample-count", "2",
                "--evaluation-set", str(gan_evaluation_bundle), "--train-limit", "2"])
    assert {path.name for path in output.glob("*.pt")} == {"best_kid.pt", "preview_inputs.pt", "evaluation_inputs.pt"}
    assert not (output / "samples_epoch_002.png").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["sampled_epochs"] == [1, 4] and summary["last_epoch"] == 4
    assert summary["best_epoch"] == 2 and summary["validation"]["g_loss"] == 1.
    for criterion, expected in {"kid": 2}.items():
        generator, discriminator, checkpoint = train.load_checkpoint(output / f"best_{criterion}.pt", torch.device("cpu"))
        assert checkpoint["epoch"] == expected and checkpoint["selection"] == summary["best_checkpoints"][criterion]
        assert torch.all(next(generator.parameters()) == expected)
        assert torch.all(next(discriminator.parameters()) == -expected)
        grid = f"samples_best_{criterion}.png"
        assert summary["best_checkpoints"][criterion]["sample_grid"] == grid
        assert (output / grid).stat().st_size > 0
        image_weights, image_noise = preview_calls[grid][-1]
        torch.testing.assert_close(image_weights, next(generator.parameters()))
        fixed = torch.load(output / "preview_inputs.pt", weights_only=True)
        torch.testing.assert_close(image_noise, fixed["noise"])
    assert len(preview_calls["samples_best_kid.png"]) == 2  # Tie at epoch 3 keeps epoch 2 and its image.
    assert summary["best_checkpoints"]["kid"]["score"] == -.2
