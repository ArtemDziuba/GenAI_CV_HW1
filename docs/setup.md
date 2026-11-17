# Repository setup

The RealNVP baseline, likelihood utilities, and training CLI are implemented;
see [the flow baseline guide](../homeworks/01_normalizing_flows/baseline.md).
AE/VAE and GAN packages remain placeholders. No dataset training results have
been recorded during implementation.
The assignment requirements remain in [readme.md](../readme.md).

## Environment

Use the existing Python requirement (3.13 or newer) and dependency constraints
in `pyproject.toml`. The `uv.lock` file records resolved versions.
From the repository root:

```sh
uv sync --locked
uv run --locked python -c "import shared, flows, vae, gan; print(shared.get_device())"
uv run --locked pytest
```

The development group includes pytest and ipykernel and is installed by default.
`uv sync` installs this project in editable mode, so imports such as
`from shared import get_device` work in notebooks as well as scripts.
Select `.venv` as the Python interpreter/kernel in your notebook editor.
For editors that need a named kernel, register one explicitly:

```sh
uv run --locked python -m ipykernel install --user --name cvhw1 --display-name "Generative Models (cvhw1)"
```

Jupyter's server application is optional; notebooks can run in an IDE using
ipykernel. No extra server or experiment-tracking dependency is required here.
Device selection prefers CUDA when PyTorch reports it available, otherwise CPU.
On Windows, the project sources torch and torchvision from the official CUDA
13.4 index rather than PyPI's CPU-only Windows builds. Other platforms retain
their existing PyPI sources. This does not install or update NVIDIA drivers.
Run `uv sync --locked` after updating the configuration/lockfile, then verify:

```sh
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The Windows build should include `+cu134`; CUDA availability also requires a
compatible NVIDIA GPU and driver. CPU execution remains available via `--device cpu`.

## Layout

```text
src/
  shared/                   # data, seeds, device selection, plotting
  flows/                    # RealNVP, likelihood, and baseline training CLI
  vae/                      # future AE/VAE and latent flow components
  gan/                      # future generators and discriminators
homeworks/
  01_normalizing_flows/      # experiments.ipynb and report.md
  02_vae/                   # experiments.ipynb and report.md
  03_gan/                   # experiments.ipynb and report.md
tests/                      # offline utility, flow correctness, and integration checks
data/                       # ignored downloads, tracked placeholder only
checkpoints/                # ignored weights, tracked placeholder only
```

Reusable training utilities and metrics belong under `src/shared/` when needed.
The flow-specific likelihood loop remains under `src/flows/`.
The pre-existing empty `src/norm_flow/` directory is unused; new flow components
belong in `src/flows/` as specified by the repository instructions.

## Shared utilities

```python
from pathlib import Path
from shared import get_device, make_dataloaders, seed_everything

seed_everything(42)
device = get_device()

# Explicit opt-in: this downloads data if it is absent. Not run by the tests.
loaders = make_dataloaders(
    "mnist",  # or "fashion_mnist" / "Fashion-MNIST"
    root=Path("data"),
    batch_size=128,
    validation_fraction=0.1,
    seed=42,
    download=True,
)
images, labels = next(iter(loaders.train))
```

Both datasets default to float32 tensors with shape `(N, 1, 28, 28)` in
`[0, 1]` and integer class labels. A custom `transform` replaces `ToTensor`.
Dequantization, likelihood corrections, binarization, and GAN normalization
must be chosen explicitly in the relevant future experiment; they are not
applied by these loaders.

The default seeded split holds out 10% of the official training set for
validation. It is random, not stratified. The official test set stays separate;
use it for final evaluation, not hyperparameter selection. Only training is
shuffled. Each loader has a local seeded RNG and retains partial final batches.
Defaults use `num_workers=0` and `pin_memory=False` for CPU/notebook portability.
On Windows, use a guarded script entry point before enabling worker processes.
The default relative data path is relative to the working directory; notebooks
resolve the repository root and pass it explicitly.

`seed_everything` covers Python, NumPy, and PyTorch. For stricter experiments,
`deterministic=True` requests deterministic algorithms and may raise errors for
unsupported operations; CUDA may require `CUBLAS_WORKSPACE_CONFIG` before launch.
Seeds do not promise identical results across platforms or library versions.

`plot_image_grid` and `plot_metrics` return matplotlib figures and accept a
`save_path` (creating parent directories). Image grids use a fixed display
range, default `[0, 1]`, so images remain comparable; use
`value_range=(-1, 1)` for outputs in that range. Call `plt.close(figure)` when
finished. Save configurations and measured metrics with future experiment
artifacts; notebooks and reports currently contain no measurements.

## Verification

Tests use synthetic tensors and replace dataset constructors; they require no
dataset downloads or training. They check imports (including from outside this
repository), CPU seeding, device dispatch, torchvision dataset selection,
preprocessing, reproducible disjoint splits and loader batches, and saved plots.
A CUDA RNG check runs only when CUDA is available. Actual dataset retrieval,
CUDA behavior on a CPU-only machine, and model correctness are not verified
by these offline checks.

If Windows denies access to pytest's default temporary directory, rerun with
`uv run --locked pytest --basetemp .pytest_cache/<fresh-directory>`, replacing
`<fresh-directory>` with a new directory name for that run.
