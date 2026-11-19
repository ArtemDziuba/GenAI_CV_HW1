"""Synthetic optimization and CLI integration; never downloads or trains MNIST."""

import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from shared.data import DataLoaders
from vae import VAE, build_model
from vae.training import importance_evaluation, run_epoch
import vae.compare as compare
import vae.evaluate as evaluation
import vae.train as cli

pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


def tiny_model(kind, device=torch.device("cpu")):
    config = dict(latent_dim=3, hidden_channels=2, hidden_dim=8)
    if kind != "ae":
        config.update(flow_steps=2 if kind == "fvae" else 0, flow_hidden_dim=8)
    return build_model(kind, **config).to(device), config


def tiny_loader(count=5):
    return DataLoader(TensorDataset(torch.zeros(count, 1, 28, 28), torch.zeros(count)), batch_size=3)


@pytest.mark.parametrize("kind", ["ae", "vae", "fvae"])
def test_synthetic_optimization_reduces_objective(kind):
    torch.manual_seed(4)
    model, _ = tiny_model(kind)
    loader = tiny_loader(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    before = run_epoch(model, loader, torch.device("cpu"), seed=10)
    for _ in range(3):
        run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, seed=10)
    after = run_epoch(model, loader, torch.device("cpu"), seed=10)
    assert after["objective"] < before["objective"]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("ae_loss, expected", [("mse", 49.), ("l1", 196.)])
def test_ae_loss_reductions_metrics_and_gradients(ae_loss, expected):
    model, _ = tiny_model("ae")
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    images = torch.full((5, 1, 28, 28), 0.25)
    loader = DataLoader(TensorDataset(images, torch.zeros(5)), batch_size=3)
    device = torch.device("cpu")
    # Zero logits reconstruct 0.5: per-pixel absolute error .25, squared .0625.
    before = run_epoch(model, loader, device, ae_loss=ae_loss)
    assert before["objective"] == pytest.approx(expected)
    assert before["grayscale_mse"] == pytest.approx(0.0625)
    assert before["grayscale_mae"] == pytest.approx(0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    run_epoch(model, loader, device, optimizer=optimizer, ae_loss=ae_loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert run_epoch(model, loader, device, ae_loss=ae_loss)["objective"] < before["objective"]


@pytest.mark.parametrize("kind", ["ae", "vae", "fvae"])
def test_evaluation_repeatable_and_no_updates(kind):
    model, _ = tiny_model(kind)
    loader = tiny_loader()
    previous = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    first = run_epoch(model, loader, torch.device("cpu"), seed=11)
    assert first == run_epoch(model, loader, torch.device("cpu"), seed=11)
    assert not model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, previous[name])
    if kind == "ae":
        with torch.no_grad():
            images = loader.dataset.tensors[0]
            assert first["grayscale_mse"] == pytest.approx((model(images) - images).square().mean().item())
            assert first["objective"] == pytest.approx(first["grayscale_mse"] * 784)


@pytest.mark.parametrize("kind", ["vae", "fvae"])
def test_grayscale_gaussian_objective_beta_and_optimizer(kind):
    model, _ = tiny_model(kind)
    images = torch.full((2, 1, 28, 28), 0.3)
    loader = DataLoader(TensorDataset(images, torch.zeros(2)), batch_size=2)
    options = dict(likelihood="gaussian", observation_std=0.2, beta=0.5, seed=18)
    before = run_epoch(model, loader, torch.device("cpu"), **options)
    assert before["negative_elbo"] == pytest.approx(before["reconstruction_nll"] + before["kl"])
    assert before["objective"] == pytest.approx(before["reconstruction_nll"] + 0.5 * before["kl"])
    optimizer = torch.optim.Adam(model.parameters())
    run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, **options)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("flow_steps", [0, 2])
def test_importance_bound_known_constant_weights(flow_steps):
    model = VAE(latent_dim=3, hidden_channels=2, hidden_dim=8, flow_steps=flow_steps, flow_hidden_dim=8)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    # q=p and decoder logits=0: all importance weights are exactly 2^-784.
    for samples in (1, 5):
        result = importance_evaluation(model, tiny_loader(), torch.device("cpu"), samples=samples)
        assert result["negative_iwae_bound"] == pytest.approx(784 * math.log(2))


@pytest.mark.parametrize("kind", ["ae", "vae", "fvae"])
def test_checkpoint_round_trip(tmp_path, kind):
    model, model_config = tiny_model(kind)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "best.pt"
    cli.save_checkpoint(path, model, optimizer, epoch=2, validation={"objective": 1.},
                        config={"kind": kind, "model": model_config})
    restored, checkpoint = cli.load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["epoch"] == 2 and not restored.training
    for name, value in restored.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])


@pytest.mark.parametrize("kind", ["ae", "ae_l1", "vae", "fvae"])
def test_cli_selection_artifacts_evaluation_and_compression(monkeypatch, tmp_path, kind):
    ae_loss = "l1" if kind == "ae_l1" else "mse"
    kind = "ae" if kind == "ae_l1" else kind
    loader = tiny_loader()
    loaders = DataLoaders(loader, loader, loader)
    monkeypatch.setattr(cli, "make_dataloaders", lambda *args, **kwargs: loaders)
    calls = []
    evaluation_values = iter([4., 6., 999.])
    def fake_epoch(model, selected_loader, device, *, optimizer=None, **kwargs):
        if kind == "ae":
            assert kwargs["ae_loss"] == ae_loss
        calls.append(optimizer is not None)
        value = 8. if optimizer is not None else next(evaluation_values)
        terms = {"objective": value, "grayscale_mse": value / 784}
        if isinstance(model, VAE):
            terms.update(reconstruction_nll=value - 1, kl=1., negative_elbo=value)
        return terms
    monkeypatch.setattr(cli, "run_epoch", fake_epoch)
    output = tmp_path / "run"
    command = ["--model", kind, "--ae-loss", ae_loss, "--device", "cpu", "--epochs", "2", "--hidden-channels", "2",
               "--hidden-dim", "8", "--latent-dim", "3", "--flow-hidden-dim", "8",
               "--flow-steps", "2", "--output", str(output), "--sample-count", "2", "--evaluate-test"]
    cli.main(command)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["best_epoch"] == 1
    assert summary["test"]["objective"] == 999
    if kind == "ae":
        config = json.loads((output / "config.json").read_text())
        assert config["objective"]["ae_loss"] == ae_loss
    assert calls == [True, False, True, False, False]
    for name in ("best.pt", "config.json", "history.json", "objective.png", "grayscale_mse.png", "reconstructions_best.png"):
        assert (output / name).stat().st_size > 0
    if kind != "ae":
        for name in ("samples_best.png", "observations_best.png", "traversal_best.png", "kl.png", "negative_elbo.png"):
            assert (output / name).stat().st_size > 0
    else:
        assert not (output / "samples_best.png").exists()
    with pytest.raises(FileExistsError):
        cli.main(command)
    # Actual synthetic evaluation through its separate CLI (still no training).
    monkeypatch.setattr(evaluation, "make_dataloaders", lambda *args, **kwargs: loaders)
    evaluated = tmp_path / "evaluated"
    evaluation.main(["--checkpoint", str(output / "best.pt"), "--device", "cpu", "--output", str(evaluated),
                     "--importance-samples", "2", "--sample-count", "2"])
    result = json.loads((evaluated / "evaluation.json").read_text())
    assert result["split"] == "validation" and result["count"] == 5
    assert math.isfinite(result["metrics"]["objective"])
    if kind == "ae":
        restored, _ = cli.load_checkpoint(output / "best.pt", torch.device("cpu"))
        expected = run_epoch(restored, loader, torch.device("cpu"), ae_loss=ae_loss)
        assert result["metrics"] == expected
    compared = tmp_path / "compared"
    compare.main(["--runs-root", str(tmp_path), "--output", str(compared)])
    assert (compared / "runs.csv").stat().st_size > 0
    if kind == "ae":
        assert (compared / "compression.png").stat().st_size > 0


@pytest.mark.parametrize("arguments", [
    ["--beta", "-1"], ["--learning-rate", "nan"], ["--conv-depth", "0"],
    ["--model", "fvae", "--latent-dim", "1"], ["--model", "fvae", "--flow-steps", "0"],
    ["--model", "ae", "--beta", "2"], ["--observation-std", "0"],
    ["--model", "vae", "--ae-loss", "l1"],
])
def test_cli_rejects_invalid_configuration(arguments):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments + ["--device", "cpu"])
    assert error.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["ae", "vae", "fvae"])
def test_cuda_synthetic_optimizer_step(kind):
    device = torch.device("cuda")
    model, _ = tiny_model(kind, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = run_epoch(model, tiny_loader(2), device, optimizer=optimizer, grad_clip=10)
    assert math.isfinite(result["objective"])
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
