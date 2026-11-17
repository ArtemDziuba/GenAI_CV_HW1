"""Synthetic optimizer/evaluation and checkpoint tests; no MNIST training runs."""

import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flows import RealNVP, RealNVPDensity
import flows.train as cli
from flows.training import run_epoch

pytestmark = pytest.mark.usefixtures("small_torch_thread_pool")


def tiny_model(device=torch.device("cpu"), *, conditioner_depth=3):
    return RealNVPDensity(RealNVP(
        (1, 4, 4), hidden_channels=4, couplings_per_scale=1,
        conditioner_depth=conditioner_depth,
    )).to(device)


def tiny_loader(count=5):
    dataset = TensorDataset(torch.zeros(count, 1, 4, 4), torch.zeros(count, dtype=torch.long))
    return DataLoader(dataset, batch_size=3, generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_one_optimizer_step_reduces_fixed_synthetic_loss(conditioner_depth):
    # Unit test on two constant synthetic images, not a dataset training run.
    torch.manual_seed(1)
    model = tiny_model(conditioner_depth=conditioner_depth)
    loader = tiny_loader(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = run_epoch(model, loader, torch.device("cpu"), seed=19)
    run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, seed=19)
    after = run_epoch(model, loader, torch.device("cpu"), seed=19)
    assert after["nll"] < before["nll"]
    assert math.isfinite(after["bpd"])


def test_evaluation_repeatability_weighting_and_no_parameter_updates():
    model = tiny_model()
    loader = tiny_loader()
    parameters = {name: value.clone() for name, value in model.state_dict().items()}
    metrics = run_epoch(model, loader, torch.device("cpu"), seed=12)
    repeated = run_epoch(model, loader, torch.device("cpu"), seed=12)
    assert metrics == repeated
    assert not model.training
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, parameters[name])
    generator = torch.Generator().manual_seed(12)
    log_prob = torch.cat([model.log_prob(images, generator=generator) for images, _ in loader])
    assert metrics["nll"] == pytest.approx(-log_prob.double().mean().item())
    assert metrics["bpd"] == pytest.approx(metrics["nll"] / (16 * math.log(2)))


@pytest.mark.parametrize("conditioner_depth", [None, 5])
def test_checkpoint_round_trip(tmp_path, conditioner_depth):
    torch.manual_seed(2)
    config = {"model": {"input_shape": (1, 4, 4), "hidden_channels": 4, "couplings_per_scale": 1},
              "preprocessing": {"logit": True, "alpha": 1e-5, "bits": 8}}
    # Missing metadata represents a checkpoint saved before depth was configurable.
    if conditioner_depth is not None:
        config["model"]["conditioner_depth"] = conditioner_depth
    depth = 3 if conditioner_depth is None else conditioner_depth
    model = tiny_model(conditioner_depth=depth)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "best.pt"
    cli.save_best_checkpoint(path, model, optimizer, epoch=2,
                             validation={"nll": 100., "bpd": 9.}, config=config)
    loaded, checkpoint = cli.load_density_checkpoint(path, torch.device("cpu"))
    assert checkpoint["epoch"] == 2 and not loaded.training
    assert loaded.flow.conditioner_depth == depth
    images, _ = next(iter(tiny_loader()))
    noise = torch.full_like(images, 0.5)
    torch.testing.assert_close(model.log_prob(images, noise=noise), loaded.log_prob(images, noise=noise))


@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_cli_checkpoint_selection_and_artifacts_without_training(
    monkeypatch, tmp_path, synthetic_dataset, conditioner_depth,
):
    # Stub epochs: verify orchestration and selection without running training.
    from shared.data import DataLoaders
    loader = DataLoader(synthetic_dataset, batch_size=4)
    monkeypatch.setattr(cli, "make_dataloaders", lambda *args, **kwargs: DataLoaders(loader, loader, loader))
    calls = []
    validation = iter([4., 6.])
    def fake_epoch(model, selected_loader, device, *, optimizer=None, seed=42, **kwargs):
        calls.append((optimizer is not None, seed))
        if optimizer is not None:
            return {"nll": 8., "bpd": 8. / (784 * math.log(2))}
        # First two evaluation calls select epoch 1; final test cannot affect it.
        value = next(validation, 999.)
        return {"nll": value, "bpd": value / (784 * math.log(2))}
    monkeypatch.setattr(cli, "run_epoch", fake_epoch)
    output = tmp_path / "run"
    cli.main(["--output", str(output), "--device", "cpu", "--epochs", "2",
              "--hidden-channels", "2", "--couplings-per-scale", "1",
              "--conditioner-depth", str(conditioner_depth),
              "--sample-count", "2", "--evaluate-test"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["best_epoch"] == 1
    assert summary["validation"]["nll"] == 4.
    assert summary["test"]["nll"] == 999.
    assert len(json.loads((output / "history.json").read_text())) == 2
    config = json.loads((output / "config.json").read_text())
    assert config["model"]["conditioner_depth"] == conditioner_depth
    assert config["arguments"]["conditioner_depth"] == conditioner_depth
    loaded, _ = cli.load_density_checkpoint(output / "best.pt", torch.device("cpu"))
    assert loaded.flow.conditioner_depth == conditioner_depth
    assert calls == [(True, 43), (False, 42), (True, 44), (False, 42), (False, 43)]
    for name in ["config.json", "best.pt", "losses.png", "samples_best.png", "samples_epoch_002.png"]:
        assert (output / name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output), "--hidden-channels", "2"])


def test_cli_conditioner_depth_default_and_validation(capsys):
    assert cli.build_parser().parse_args([]).conditioner_depth == 3
    with pytest.raises(SystemExit) as error:
        cli.main(["--conditioner-depth", "2"])
    assert error.value.code == 2
    assert "--conditioner-depth must be at least 3" in capsys.readouterr().err


def test_explicit_cuda_fails_without_available_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        cli.resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("conditioner_depth", [3, 5])
def test_cuda_forward_backward_optimizer_and_inverse(conditioner_depth):
    device = torch.device("cuda")
    model = tiny_model(device, conditioner_depth=conditioner_depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, tiny_loader(2), device, optimizer=optimizer, seed=3)
    assert math.isfinite(metrics["nll"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    inputs = torch.randn(2, 1, 4, 4, device=device)
    latents, _ = model.flow(inputs)
    recovered, _ = model.flow.inverse(latents)
    torch.testing.assert_close(recovered, inputs, atol=1e-5, rtol=1e-5)
