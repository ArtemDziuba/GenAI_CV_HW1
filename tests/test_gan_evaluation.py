"""Frozen classifier provenance, split isolation, input banks and sample counts."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from shared.data import DataLoaders
from gan.classifier import load_classifier
from gan.evaluation import GeneratorEvaluator
from gan.models import Generator
import gan.prepare_evaluation as prepare
import gan.train_classifier as classifier_training
import gan.train as train


pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


@pytest.mark.parametrize("conditional", [False, True])
def test_fixed_evaluation_is_repeatable_frozen_and_uses_exact_counts(gan_evaluation_bundle, conditional):
    evaluator = GeneratorEvaluator(gan_evaluation_bundle, torch.device("cpu"), latent_dim=4,
                                   conditional=conditional, batch_size=7, coverage_min_count=1)
    generator = Generator(latent_dim=4, channels=2, conditional=conditional)
    before = {name: value.clone() for name, value in generator.state_dict().items()}
    classifier_before = {name: value.clone() for name, value in evaluator.classifier.state_dict().items()}
    rng = torch.random.get_rng_state()
    metrics = evaluator.evaluate(generator)
    assert metrics == evaluator.evaluate(generator)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert metrics["samples"] == metrics["real_samples"] == 20  # Includes final partial batch.
    assert sum(metrics["class_counts"]) == 20
    assert generator.training
    assert all(torch.equal(value, generator.state_dict()[name]) for name, value in before.items())
    assert all(torch.equal(value, evaluator.classifier.state_dict()[name]) for name, value in classifier_before.items())
    assert not evaluator.classifier.training and not any(p.requires_grad for p in evaluator.classifier.parameters())
    repeated = GeneratorEvaluator(gan_evaluation_bundle, torch.device("cpu"), latent_dim=4, conditional=conditional)
    torch.testing.assert_close(evaluator.noise, repeated.noise)
    if conditional:
        reference = torch.load(gan_evaluation_bundle / "reference.pt", weights_only=True)
        torch.testing.assert_close(evaluator.labels, reference["labels"])
        assert "conditional_accuracy" in metrics


def test_reference_file_tampering_is_rejected(gan_evaluation_bundle):
    path = gan_evaluation_bundle / "reference.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bundle changed"):
        GeneratorEvaluator(gan_evaluation_bundle, torch.device("cpu"), latent_dim=4, conditional=False)


@pytest.mark.parametrize("count", [20, 5001])
def test_full_training_rejects_smoke_reference_bundle(tmp_path, gan_evaluation_bundle, monkeypatch, count):
    original = train.GeneratorEvaluator
    def changed_count(*args, **kwargs):
        evaluator = original(*args, **kwargs)
        evaluator.count = count
        return evaluator
    monkeypatch.setattr(train, "GeneratorEvaluator", changed_count)
    output = tmp_path / "must_not_train"
    with pytest.raises(SystemExit):
        train.main(["--evaluation-set", str(gan_evaluation_bundle), "--device", "cpu", "--output", str(output)])
    assert not output.exists()


def test_training_rejects_cross_dataset_reference(tmp_path, gan_evaluation_bundle):
    with pytest.raises(SystemExit):
        train.main(["--evaluation-set", str(gan_evaluation_bundle), "--dataset", "fashion_mnist", "--train-limit", "2",
                    "--device", "cpu", "--output", str(tmp_path / "wrong_dataset")])


def test_classifier_fit_and_internal_selection_exclude_reference_images(monkeypatch, tmp_path, synthetic_dataset):
    train_data = Subset(synthetic_dataset, range(30))
    reference_data = Subset(synthetic_dataset, range(30, 40))
    loaders = DataLoaders(DataLoader(train_data, batch_size=8), DataLoader(reference_data, batch_size=8),
                         DataLoader(reference_data, batch_size=8))
    monkeypatch.setattr(classifier_training, "make_dataloaders", lambda *args, **kwargs: loaders)
    monkeypatch.setattr(prepare, "make_dataloaders", lambda *args, **kwargs: loaders)
    seen = set()
    original_epoch = classifier_training.classifier_epoch
    def track(model, loader, device, optimizer=None):
        dataset = loader.dataset
        assert isinstance(dataset, Subset) and dataset.dataset is train_data
        seen.update(int(train_data.indices[index]) for index in dataset.indices)
        return original_epoch(model, loader, device, optimizer)
    monkeypatch.setattr(classifier_training, "classifier_epoch", track)
    output = tmp_path / "classifier"
    classifier_training.main(["--dataset", "mnist", "--device", "cpu", "--output", str(output), "--epochs", "1",
                              "--channels", "2", "--feature-dim", "4", "--batch-size", "8", "--classifier-validation-fraction", ".2"])
    assert seen == set(range(30))
    assert seen.isdisjoint(reference_data.indices)
    frozen, checkpoint = load_classifier(output / "best.pt", torch.device("cpu"))
    assert not any(p.requires_grad for p in frozen.parameters())
    assert checkpoint["config"]["fit_images"] == 24 and checkpoint["config"]["selection_images"] == 6
    bundle = tmp_path / "prepared"
    # Synthetic classifier quality is deliberately not used as a scientific result.
    prepare.main(["--classifier", str(output / "best.pt"), "--output", str(bundle), "--device", "cpu",
                  "--count", "8", "--min-classifier-accuracy", "0"])
    cache = torch.load(bundle / "reference.pt", weights_only=True)
    assert len(cache["features"]) == len(cache["images"]) == 8
    assert set(cache["source_indices"].tolist()).issubset(set(range(30, 40)))
    assert seen.isdisjoint(cache["source_indices"].tolist())
    assert json.loads((bundle / "config.json").read_text())["split"] == "validation"
    # The regular quality gate must reject an obviously unrecognizing classifier.
    original_load = prepare.load_classifier
    def unrecognizing(path, device):
        model, saved = original_load(path, device)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        return model, saved
    monkeypatch.setattr(prepare, "load_classifier", unrecognizing)
    with pytest.raises(ValueError, match="below"):
        prepare.main(["--classifier", str(output / "best.pt"), "--output", str(tmp_path / "weak"),
                      "--device", "cpu", "--count", "8"])
    assert not (tmp_path / "weak").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_classifier_evaluation(gan_evaluation_bundle):
    evaluator = GeneratorEvaluator(gan_evaluation_bundle, torch.device("cuda"), latent_dim=4,
                                   conditional=True, batch_size=8)
    generator = Generator(latent_dim=4, channels=2, conditional=True).cuda()
    metrics = evaluator.evaluate(generator)
    assert metrics["samples"] == 20 and torch.isfinite(torch.tensor(metrics["kid"]))
