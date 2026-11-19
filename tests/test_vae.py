"""Model, posterior-density, and observation-likelihood correctness, offline."""

import math

import pytest
import torch
from torch import nn

from vae import (
    Autoencoder, LatentFlow, VAE, build_model, diagonal_normal_log_prob,
    prepare_observations, reconstruction_nll, reparameterize, standard_normal_kl, vae_terms,
)

pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


@pytest.mark.parametrize("kind", ["ae", "vae", "fvae"])
@pytest.mark.parametrize("depth", [1, 2])
def test_model_shapes_and_gradient_paths(kind, depth):
    config = dict(latent_dim=7, hidden_channels=2, hidden_dim=8, conv_depth=depth)
    if kind != "ae":
        config.update(flow_steps=2 if kind == "fvae" else 0, flow_hidden_dim=8)
    model = build_model(kind, **config)
    images = torch.zeros(3, 1, 28, 28)
    if kind == "ae":
        reconstructed = model(images)
        loss = (reconstructed - images).square().sum(1).mean()
        assert model.encode(images).shape == (3, 7)
    else:
        output = model(images)
        reconstructed = output.logits.sigmoid()
        assert output.mean.shape == output.log_variance.shape == output.latent.shape == (3, 7)
        assert output.log_q.shape == output.log_prior.shape == output.log_det.shape == (3,)
        loss = vae_terms(output, images, has_flow=kind == "fvae")["objective"].mean()
    assert reconstructed.shape == images.shape
    assert torch.isfinite(reconstructed).all()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if kind != "ae":
        assert model.mean_head.weight.grad.abs().sum() > 0
        assert model.log_variance_head.weight.grad.abs().sum() > 0


def test_reparameterization_values_and_gradients():
    mean = torch.tensor([[1., -2.]], requires_grad=True)
    log_variance = torch.tensor([[0., math.log(4)]], requires_grad=True)
    expected_noise = torch.randn(1, 2, generator=torch.Generator().manual_seed(19))
    latent = reparameterize(mean, log_variance, generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(latent, mean + expected_noise * torch.tensor([[1., 2.]]))
    latent.sum().backward()
    torch.testing.assert_close(mean.grad, torch.ones_like(mean))
    torch.testing.assert_close(log_variance.grad, 0.5 * expected_noise * torch.tensor([[1., 2.]]))


def test_gaussian_kl_and_log_density_match_torch_distributions():
    mean = torch.tensor([[0., 0.], [1., -2.]])
    log_variance = torch.tensor([[0., 0.], [math.log(2), math.log(0.5)]])
    q = torch.distributions.Normal(mean, (0.5 * log_variance).exp())
    prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    torch.testing.assert_close(standard_normal_kl(mean, log_variance),
                               torch.distributions.kl_divergence(q, prior).sum(1))
    assert standard_normal_kl(mean, log_variance)[0] == 0
    value = torch.ones_like(mean)
    torch.testing.assert_close(diagonal_normal_log_prob(value, mean, log_variance), q.log_prob(value).sum(1))


def nonidentity_flow(dim=3):
    torch.manual_seed(17)
    flow = LatentFlow(dim, steps=3, hidden_dim=8).double()
    for layer in flow.layers:
        nn.init.normal_(layer.conditioner[-1].weight, std=0.08)
        nn.init.normal_(layer.conditioner[-1].bias, std=0.08)
    return flow


@pytest.mark.parametrize("dim", [2, 3, 7])
def test_latent_flow_inverse_masks_and_jacobian(dim):
    flow = nonidentity_flow(dim)
    value = torch.randn(2, dim, dtype=torch.float64)
    transformed, log_det = flow(value)
    recovered, inverse_log_det = flow.inverse(transformed)
    torch.testing.assert_close(recovered, value)
    torch.testing.assert_close(log_det + inverse_log_det, torch.zeros(2, dtype=value.dtype))
    jacobian = torch.autograd.functional.jacobian(lambda x: flow(x[None])[0][0], value[0])
    sign, numerical = torch.linalg.slogdet(jacobian)
    assert sign == 1
    torch.testing.assert_close(log_det[0], numerical)
    assert not torch.equal(transformed, value)


def test_flow_posterior_density_uses_negative_forward_log_det():
    model = VAE(latent_dim=3, hidden_channels=2, hidden_dim=8, flow_steps=3, flow_hidden_dim=8).double()
    model.posterior_flow = nonidentity_flow()
    output = model(torch.zeros(2, 1, 28, 28, dtype=torch.float64))
    expected = diagonal_normal_log_prob(output.base_latent, output.mean, output.log_variance) - output.log_det
    torch.testing.assert_close(output.log_q, expected)
    terms = vae_terms(output, torch.zeros_like(output.logits), has_flow=True, beta=2)
    torch.testing.assert_close(terms["kl"], expected - output.log_prior)
    torch.testing.assert_close(terms["negative_elbo"], terms["reconstruction_nll"] + terms["kl"])
    torch.testing.assert_close(terms["objective"], terms["negative_elbo"] + terms["kl"])


def test_sampled_flow_kl_is_not_clamped():
    model = VAE(latent_dim=2, hidden_channels=2, hidden_dim=8)
    output = model(torch.zeros(1, 1, 28, 28))
    output.log_q = torch.tensor([-2.])
    output.log_prior = torch.tensor([-1.])
    assert vae_terms(output, torch.zeros_like(output.logits), has_flow=True)["kl"].item() == -1


def test_prior_sampling_bypasses_posterior_flow(monkeypatch):
    model = VAE(latent_dim=3, hidden_channels=2, hidden_dim=8, flow_steps=2)
    def forbidden(*args, **kwargs):
        raise AssertionError("prior sampling must not use the posterior flow")
    monkeypatch.setattr(model.posterior_flow, "forward", forbidden)
    latent = torch.randn(4, 3, generator=torch.Generator().manual_seed(8))
    expected = model.decode(latent)
    sampled = model.sample(4, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(sampled, expected)
    assert not sampled.requires_grad


def test_binarization_and_observation_likelihood_constants():
    images = torch.full((2, 1, 28, 28), 0.3)
    observed = prepare_observations(images, "bernoulli", generator=torch.Generator().manual_seed(9))
    repeated = prepare_observations(images, "bernoulli", generator=torch.Generator().manual_seed(9))
    assert torch.equal(observed, repeated)
    assert torch.all((observed == 0) | (observed == 1))
    assert prepare_observations(images, "gaussian") is images
    logits = torch.zeros_like(images)
    torch.testing.assert_close(reconstruction_nll(logits, observed), torch.full((2,), 784 * math.log(2)))
    with pytest.raises(ValueError, match="binary"):
        reconstruction_nll(logits, images)
    gaussian = reconstruction_nll(logits, images, likelihood="gaussian", observation_std=0.2)
    expected = -torch.distributions.Normal(torch.full_like(images, 0.5), 0.2).log_prob(images).flatten(1).sum(1)
    torch.testing.assert_close(gaussian, expected)
    with pytest.raises(ValueError):
        reconstruction_nll(logits, images, likelihood="gaussian", observation_std=0)


def test_architecture_validation():
    with pytest.raises(ValueError):
        Autoencoder(latent_dim=0)
    with pytest.raises(ValueError, match="shape"):
        Autoencoder()(torch.zeros(2, 1, 32, 32))
    with pytest.raises(ValueError):
        LatentFlow(1, steps=2)
    with pytest.raises(ValueError):
        build_model("fvae", flow_steps=0)
    with pytest.raises(ValueError):
        build_model("vae", flow_steps=2)
