"""Mathematical flow checks: nontrivial transforms, Jacobians, and densities."""

import math

import pytest
import torch
from torch import nn

from flows import (
    AffineCoupling, CouplingNetwork, ImagePreprocessing, RealNVP, RealNVPDensity,
    bits_per_dimension, channel_mask, checkerboard_mask, negative_log_likelihood,
    squeeze2d, standard_normal_log_prob, uniform_dequantize, unsqueeze2d,
)

pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


def make_nontrivial(flow):
    # Identity initialization alone would not exercise inverse ordering or signs.
    with torch.no_grad():
        for layer in flow.modules():
            if isinstance(layer, CouplingNetwork):
                nn.init.normal_(layer.net[-1].weight, std=0.015)
                nn.init.normal_(layer.net[-1].bias, std=0.015)
    return flow


def test_masks():
    checker = checkerboard_mask(3, 4)
    assert checker.shape == (1, 1, 3, 4)
    torch.testing.assert_close(checker[0, 0, 0], torch.tensor([0., 1., 0., 1.]))
    torch.testing.assert_close(checker + checkerboard_mask(3, 4, invert=True), torch.ones_like(checker))
    channel = channel_mask(4)
    torch.testing.assert_close(channel.flatten(), torch.tensor([1., 1., 0., 0.]))
    torch.testing.assert_close(channel + channel_mask(4, invert=True), torch.ones_like(channel))


@pytest.mark.parametrize("mask", [checkerboard_mask(4, 4), channel_mask(4)])
@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_image_coupling_inverse_and_gradients(mask, conditioner_depth):
    torch.manual_seed(1)
    coupling = AffineCoupling(
        CouplingNetwork(4, 5, conditioner_depth=conditioner_depth), mask
    ).double()
    with torch.no_grad():
        nn.init.normal_(coupling.conditioner.net[-1].weight, std=0.03)
    inputs = torch.randn(2, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    outputs, log_det = coupling(inputs)
    recovered, inverse_log_det = coupling.inverse(outputs)
    torch.testing.assert_close(recovered, inputs, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(log_det + inverse_log_det, torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(outputs * mask, inputs * mask)
    assert not torch.allclose(outputs, inputs)
    (outputs.square().mean() + log_det.mean()).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in coupling.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in coupling.parameters())


def test_coupling_jacobian_against_finite_differences():
    torch.manual_seed(2)
    coupling = AffineCoupling(
        nn.Linear(4, 8).double(), torch.tensor([[1., 0., 1., 0.]]), scale_limit=0.7
    ).double()
    inputs = torch.randn(4, dtype=torch.float64)
    def transform(vector):
        return coupling(vector.unsqueeze(0))[0].squeeze(0)
    epsilon = 1e-6
    identity = torch.eye(4, dtype=torch.float64)
    jacobian = torch.stack([
        (transform(inputs + epsilon * direction) - transform(inputs - epsilon * direction)) / (2 * epsilon)
        for direction in identity
    ], dim=1)
    analytical = coupling(inputs.unsqueeze(0))[1][0]
    torch.testing.assert_close(torch.linalg.slogdet(jacobian).logabsdet, analytical, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_zero_initialized_conditioner_starts_at_identity(conditioner_depth):
    coupling = AffineCoupling(
        CouplingNetwork(1, 4, conditioner_depth=conditioner_depth), checkerboard_mask(4, 4)
    )
    inputs = torch.randn(2, 1, 4, 4)
    outputs, log_det = coupling(inputs)
    torch.testing.assert_close(outputs, inputs, atol=0, rtol=0)
    torch.testing.assert_close(log_det, torch.zeros(2), atol=0, rtol=0)


@pytest.mark.parametrize("conditioner_depth", [3, 4, 5])
def test_conditioner_depth_expands_spatial_context(conditioner_depth):
    network = CouplingNetwork(1, 1, conditioner_depth=conditioner_depth)
    # Positive weights make every reachable pixel respond to a central impulse.
    with torch.no_grad():
        for layer in network.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    inputs = torch.zeros(1, 1, 11, 11)
    inputs[0, 0, 5, 5] = 1
    outputs = network(inputs)
    assert outputs.shape == (1, 2, 11, 11)
    assert sum(isinstance(layer, nn.Conv2d) for layer in network.net) == conditioner_depth
    radius = conditioner_depth - 1
    expected_support = torch.zeros(11, 11, dtype=torch.bool)
    expected_support[5-radius:6+radius, 5-radius:6+radius] = True
    assert torch.equal(outputs[0, 0] > 0, expected_support)


def test_default_conditioner_loads_legacy_sequential_weights():
    # The exact former three-convolution layout: saved parameter names must match.
    legacy = nn.Sequential(
        nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 4, 1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1),
    )
    current = CouplingNetwork(2, 4)
    current.net.load_state_dict(legacy.state_dict(), strict=True)
    inputs = torch.randn(2, 2, 4, 4)
    torch.testing.assert_close(current(inputs), legacy(inputs), atol=0, rtol=0)


@pytest.mark.parametrize("conditioner_depth", [0, 2, 3.5, True])
def test_invalid_conditioner_depth(conditioner_depth):
    with pytest.raises(ValueError, match="conditioner_depth"):
        CouplingNetwork(1, 4, conditioner_depth=conditioner_depth)
    with pytest.raises(ValueError, match="conditioner_depth"):
        RealNVP(conditioner_depth=conditioner_depth)


def test_squeeze_is_an_exact_permutation():
    images = torch.arange(2 * 3 * 8 * 6).reshape(2, 3, 8, 6)
    squeezed = squeeze2d(images)
    assert squeezed.shape == (2, 12, 4, 3)
    assert torch.equal(unsqueeze2d(squeezed), images)
    assert torch.equal(squeezed[:, 0], images[:, 0, 0::2, 0::2])
    assert torch.equal(squeezed[:, 1], images[:, 0, 0::2, 1::2])
    assert torch.equal(squeezed[:, 2], images[:, 0, 1::2, 0::2])


@pytest.mark.parametrize("shape, scales", [((1, 28, 28), 2), ((3, 8, 12), 2), ((2, 8, 8), 3)])
@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_complete_multiscale_inverse_and_dimensions(shape, scales, conditioner_depth):
    torch.manual_seed(3)
    flow = make_nontrivial(RealNVP(shape, num_scales=scales, hidden_channels=4,
                                  couplings_per_scale=2,
                                  conditioner_depth=conditioner_depth).double())
    inputs = torch.randn(2, *shape, dtype=torch.float64)
    latents, log_det = flow(inputs)
    recovered, inverse_log_det = flow.inverse(latents)
    assert latents.shape == (2, math.prod(shape))
    assert sum(flow.latent_sizes) == math.prod(shape)
    if shape == (1, 28, 28):
        assert flow.latent_sizes == [392, 392]
    torch.testing.assert_close(recovered, inputs, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-9, rtol=0)


def test_latent_block_order_at_identity():
    flow = RealNVP(hidden_channels=2, couplings_per_scale=1)
    images = torch.arange(784, dtype=torch.float32).reshape(1, 1, 28, 28)
    active, factored = squeeze2d(images).chunk(2, dim=1)
    expected = torch.cat((factored.flatten(1), squeeze2d(active).flatten(1)), dim=1)
    latents, log_det = flow(images)
    assert torch.equal(latents, expected)
    assert torch.equal(log_det, torch.zeros(1))


@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_full_multiscale_jacobian(conditioner_depth):
    torch.manual_seed(4)
    flow = make_nontrivial(RealNVP(
        (1, 4, 4), hidden_channels=3, couplings_per_scale=1,
        conditioner_depth=conditioner_depth,
    ).double())
    inputs = torch.randn(16, dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        lambda vector: flow(vector.reshape(1, 1, 4, 4))[0].flatten(), inputs
    )
    actual = torch.linalg.slogdet(jacobian).logabsdet
    expected = flow(inputs.reshape(1, 1, 4, 4))[1][0]
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("shape, scales", [((1, 28, 28), 3), ((1, 7, 8), 1)])
def test_invalid_multiscale_dimensions(shape, scales):
    with pytest.raises(ValueError, match="divisible"):
        RealNVP(shape, num_scales=scales)


def test_dequantization_matches_8bit_bins():
    images = torch.tensor([0., 1 / 255, 128 / 255, 1.]).reshape(1, 1, 2, 2)
    noise = torch.full_like(images, 0.25)
    result = uniform_dequantize(images, noise=noise)
    expected = (torch.tensor([0., 1., 128., 255.]) + 0.25).reshape_as(images) / 256
    torch.testing.assert_close(result, expected)
    assert result.min() >= 0 and result.max() < 1
    first = uniform_dequantize(images, generator=torch.Generator().manual_seed(9))
    second = uniform_dequantize(images, generator=torch.Generator().manual_seed(9))
    assert torch.equal(first, second)


@pytest.mark.parametrize("logit", [True, False])
def test_preprocessing_inverse_and_jacobian(logit):
    preprocessing = ImagePreprocessing(logit=logit, alpha=0.05)
    inputs = torch.tensor([0., 0.2, 0.7, 1.], dtype=torch.float64).reshape(1, 1, 2, 2)
    outputs, log_det = preprocessing(inputs)
    recovered, inverse_log_det = preprocessing.inverse(outputs)
    torch.testing.assert_close(recovered, inputs, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(log_det + inverse_log_det, torch.zeros_like(log_det), atol=1e-12, rtol=0)
    jacobian = torch.autograd.functional.jacobian(
        lambda vector: preprocessing(vector.reshape_as(inputs))[0].flatten(), inputs.flatten()
    )
    torch.testing.assert_close(torch.linalg.slogdet(jacobian).logabsdet, log_det[0])


@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_joint_preprocessing_and_flow_likelihood_jacobian(conditioner_depth):
    torch.manual_seed(5)
    model = RealNVPDensity(make_nontrivial(RealNVP(
        (1, 2, 2), num_scales=1, hidden_channels=3, couplings_per_scale=1,
        conditioner_depth=conditioner_depth,
    ).double()), alpha=0.03)
    images = torch.tensor([0., 30., 128., 255.], dtype=torch.float64).reshape(1, 1, 2, 2) / 255
    noise = torch.full_like(images, 0.25)
    unit = (torch.round(images * 255) + noise) / 256
    def transform(vector):
        transformed, _ = model.preprocessing(vector.reshape_as(images))
        return model.flow(transformed)[0].flatten()
    jacobian = torch.autograd.functional.jacobian(transform, unit.flatten())
    latents = transform(unit.flatten()).unsqueeze(0)
    expected = standard_normal_log_prob(latents) + torch.linalg.slogdet(jacobian).logabsdet - 4 * math.log(256)
    actual = model.log_prob(images, noise=noise)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_gaussian_constant_and_bpd_conventions():
    normal = standard_normal_log_prob(torch.zeros(2, 784))
    torch.testing.assert_close(normal, torch.full((2,), -392 * math.log(2 * math.pi)))
    log_prob = torch.full((3,), -784 * math.log(256), dtype=torch.float64)
    torch.testing.assert_close(bits_per_dimension(log_prob, 784), torch.full_like(log_prob, 8))
    torch.testing.assert_close(negative_log_likelihood(log_prob), -log_prob.mean())


@pytest.mark.parametrize("logit", [True, False])
@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_finite_extreme_images_and_parameter_gradients(logit, conditioner_depth):
    model = RealNVPDensity(RealNVP(
        hidden_channels=3, couplings_per_scale=1, conditioner_depth=conditioner_depth,
    ), logit=logit)
    images = torch.stack((torch.zeros(1, 28, 28), torch.ones(1, 28, 28)))
    log_prob = model.log_prob(images, noise=torch.zeros_like(images))
    assert log_prob.shape == (2,) and torch.isfinite(log_prob).all()
    negative_log_likelihood(log_prob).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    samples = model.sample(3, generator=torch.Generator().manual_seed(6))
    assert samples.shape == (3, 1, 28, 28) and torch.isfinite(samples).all()


def test_preprocessing_inverse_does_not_clip():
    preprocessing = ImagePreprocessing(alpha=0.05)
    outputs = torch.tensor([[-20., 20.]])
    inputs, _ = preprocessing.inverse(outputs)
    assert inputs[0, 0] < 0 and inputs[0, 1] > 1
