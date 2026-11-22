"""Offline GAN math, gradient isolation, conditioning and optimization checks."""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gan import Discriminator, Generator, discriminator_step, generator_step, run_epoch
from gan.artifacts import fixed_inputs, save_samples
from gan.training import discriminator_terms, feature_matching


pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


def models(conditional=False, head="global", device="cpu"):
    return (Generator(4, 2, conditional, 3).to(device), Discriminator(2, conditional, head).to(device))


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("head", ["global", "patch"])
def test_shapes_finite_outputs_and_conditional_gradients(conditional, head):
    generator, discriminator = models(conditional, head)
    labels = torch.tensor([1, 8, 3]) if conditional else None
    fake = generator(torch.randn(3, 4), labels)
    assert fake.shape == (3, 1, 28, 28)
    assert torch.isfinite(fake).all() and fake.min() >= -1 and fake.max() <= 1
    logits, features = discriminator(fake, labels, return_features=True)
    assert logits.shape == ((3, 1) if head == "global" else (3, 1, 7, 7))
    assert features.shape == (3, 4, 7, 7)
    logits.sum().backward()
    for module in (generator, discriminator):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    if conditional:
        for module in (generator, discriminator):
            assert module.label_embedding.weight.grad[[1, 8, 3]].abs().sum() > 0
            assert module.label_embedding.weight.grad[0].abs().sum() == 0


@pytest.mark.parametrize("labels", [None, torch.tensor([0., 1.]), torch.tensor([0]), torch.tensor([-1, 0]), torch.tensor([0, 10])])
def test_invalid_conditional_labels(labels):
    generator, discriminator = models(True)
    with pytest.raises(ValueError):
        generator(torch.randn(2, 4), labels)
    with pytest.raises(ValueError):
        discriminator(torch.zeros(2, 1, 28, 28), labels)


def test_unconditional_models_reject_labels():
    generator, discriminator = models()
    with pytest.raises(ValueError):
        generator(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        discriminator(torch.zeros(2, 1, 28, 28), torch.zeros(2, dtype=torch.long))


@pytest.mark.parametrize("shape", [(3, 1), (3, 1, 7, 7)])
def test_bce_reductions_and_numerical_stability(shape):
    terms = discriminator_terms(torch.zeros(shape), torch.zeros(shape))
    assert terms["d_loss"].item() == pytest.approx(math.log(2))
    assert terms["d_real_probability"].item() == .5
    assert terms["d_fake_probability"].item() == .5
    # Extreme correct logits are stable, including PatchGAN's many logits.
    good = discriminator_terms(torch.full(shape, 1000.), torch.full(shape, -1000.))
    assert good["d_loss"].item() == 0 and good["d_accuracy"].item() == 1
    bad = discriminator_terms(torch.full(shape, -1000.), torch.full(shape, 1000.))
    assert bad["d_loss"].item() == 1000 and torch.isfinite(bad["d_loss"])


def test_feature_matching_is_batch_mean_and_stops_real_target_gradient():
    real = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    fake = torch.tensor([[0., 1.], [2., 1.]], requires_grad=True)
    loss = feature_matching(real, fake)
    assert loss.item() == pytest.approx(2.5)  # mean(([2,3] - [1,1])**2)
    loss.backward()
    assert real.grad is None and torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0


@pytest.mark.parametrize("conditional,head,weight", [(False, "global", 0.), (True, "global", 0.),
                                                    (False, "patch", 0.), (True, "patch", 10.)])
def test_independent_optimizer_steps_preserve_frozen_parameters_and_buffers(conditional, head, weight):
    generator, discriminator = models(conditional, head)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.001)
    real = torch.rand(3, 1, 28, 28) * 2 - 1
    labels = torch.tensor([0, 2, 7]) if conditional else None
    original_g = {key: value.clone() for key, value in generator.state_dict().items()}
    original_d = {key: value.clone() for key, value in discriminator.state_dict().items()}
    metrics = discriminator_step(generator, discriminator, real, labels, torch.randn(3, 4), d_optimizer)
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(torch.equal(value, generator.state_dict()[key]) for key, value in original_g.items())
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert any(not torch.equal(value, discriminator.state_dict()[key]) for key, value in original_d.items())
    updated_d = {key: value.clone() for key, value in discriminator.state_dict().items()}
    metrics = generator_step(generator, discriminator, real, labels, torch.randn(3, 4), g_optimizer,
                             feature_matching_weight=weight)
    assert metrics["g_loss"] == pytest.approx(metrics["g_adversarial_loss"] + weight * metrics["g_feature_matching_loss"])
    assert all(torch.equal(value, discriminator.state_dict()[key]) for key, value in updated_d.items())
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in generator.parameters())
    assert any(not torch.equal(value, generator.state_dict()[key]) for key, value in original_g.items())


def test_repeatable_validation_no_updates_and_partial_batch_reduction():
    generator, discriminator = models(True, "patch")
    loader = DataLoader(TensorDataset(torch.rand(5, 1, 28, 28), torch.arange(5)), batch_size=3)
    originals = [{key: value.clone() for key, value in model.state_dict().items()} for model in (generator, discriminator)]
    batches = []
    result = run_epoch(generator, discriminator, loader, torch.device("cpu"), feature_matching_weight=10,
                       on_step=lambda index, metrics: batches.append(metrics))
    assert result == run_epoch(generator, discriminator, loader, torch.device("cpu"), feature_matching_weight=10)
    assert result["images"] == 5 and result["batches"] == 2 and result["g_updates"] == result["d_updates"] == 0
    assert result["g_loss"] == pytest.approx(sum(row["g_loss"] * row["batch_images"] for row in batches) / 5)
    for model, original in zip((generator, discriminator), originals):
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())


def test_training_epoch_update_counts_and_finite_losses():
    generator, discriminator = models()
    loader = DataLoader(TensorDataset(torch.rand(5, 1, 28, 28), torch.zeros(5)), batch_size=2)
    result = run_epoch(generator, discriminator, loader, torch.device("cpu"),
                       generator_optimizer=torch.optim.Adam(generator.parameters()),
                       discriminator_optimizer=torch.optim.Adam(discriminator.parameters()), discriminator_steps=2)
    assert result["g_updates"] == 3 and result["d_updates"] == 6 and result["images"] == 5
    assert all(math.isfinite(value) for value in result.values())


def test_fixed_conditional_previews_preserve_batchnorm_and_training_rng(tmp_path):
    generator, _ = models(True)
    original = {key: value.clone() for key, value in generator.state_dict().items()}
    rng_before = torch.random.get_rng_state()
    noise, labels = fixed_inputs(generator, samples_per_class=2)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(labels, torch.arange(10).repeat_interleave(2))
    diagnostics = save_samples(generator, noise, labels, tmp_path / "preview.png", columns=2)
    assert generator.training
    assert all(torch.equal(value, generator.state_dict()[key]) for key, value in original.items())
    assert (tmp_path / "preview.png").stat().st_size > 0
    assert all(math.isfinite(value) and value >= 0 for value in diagnostics.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("conditional,head,weight", [(False, "global", 0.), (True, "global", 0.),
                                                    (False, "patch", 0.), (True, "patch", 10.)])
def test_cuda_optimizer_steps(conditional, head, weight):
    generator, discriminator = models(conditional, head, "cuda")
    real = torch.rand(2, 1, 28, 28, device="cuda") * 2 - 1
    labels = torch.tensor([1, 8], device="cuda") if conditional else None
    noise = torch.randn(2, 4, device="cuda")
    d = discriminator_step(generator, discriminator, real, labels, noise, torch.optim.Adam(discriminator.parameters()))
    g = generator_step(generator, discriminator, real, labels, noise, torch.optim.Adam(generator.parameters()),
                       feature_matching_weight=weight)
    assert all(math.isfinite(value) for value in {**d, **g}.values())
