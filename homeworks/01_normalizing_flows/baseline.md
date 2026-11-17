# RealNVP baseline

Implementation is ready to run. No MNIST training or dataset downloads were
performed during implementation. Initial verification used CPU-only PyTorch.
The Windows environment now uses `2.15.0+cu134`; GPU detection, a GPU matrix
operation, and the synthetic CUDA forward/backward and inverse test pass on
the RTX 4060 Laptop GPU. The student has since trained the MNIST baseline;
further training commands are also run by the student.

## Run from the repository root

Install the existing locked environment if necessary with `uv sync --locked`.
The GPU commands require a CUDA-enabled PyTorch installation and an available
GPU. `--device cuda` raises an error when CUDA is unavailable instead of silently
falling back to CPU. `--device auto` allows the shared CUDA/CPU selection policy.

First run a small-subset smoke experiment yourself:

```sh
uv run --locked python -m flows.train --dataset mnist --device cuda --download --epochs 3 --batch-size 64 --train-limit 1024 --validation-limit 256 --output checkpoints/realnvp_mnist_smoke
```

Inspect `history.json` and `losses.png` for finite losses and learning on this
subset. A decrease on synthetic unit-test data does not establish a decrease
on actual MNIST, and individual stochastic epochs need not decrease monotonically.
Then run the baseline:

```sh
uv run --locked python -m flows.train --dataset mnist --device cuda --download --epochs 30 --batch-size 64 --hidden-channels 64 --num-scales 2 --couplings-per-scale 3 --learning-rate 0.001 --seed 42 --evaluate-test --output checkpoints/realnvp_mnist_baseline
```

Use a new output directory for every invocation; existing paths are refused.
Omit `--download` when the data is already present. No process starts on import.
For Fashion-MNIST, change `--dataset fashion_mnist` and the output directory.
Thirty epochs is the standard budget for the architecture comparisons below.
Use `--help` to inspect all options, including `--no-logit` and `--logit-alpha`.
Keep `--num-workers 0` for the simplest Windows setup; the CLI has a guarded
entry point for enabling worker processes later.

## Current architecture experiment plan

The student reports that the 60-epoch baseline overfits. Keep all subsequent
architecture comparisons at 30 epochs. This is a student-reported observation;
the 60-epoch learning curves have not been independently reviewed here.

Run the following experiments in order:

| Experiment | Hidden channels | Couplings per mask family per scale | Total couplings | Epochs |
| --- | ---: | ---: | ---: | ---: |
| Existing baseline | 64 | 3 | 12 | 30 |
| 1. Wider | 96 | 3 | 12 | 30 |
| 2. More transformations | 64 | 4 | 16 | 30 |
| 3. Wider and more transformations | 96 | 4 | 16 | 30 |

Keep learning rate 0.001, batch size 64, seed 42, two scales, the same data
split, and the same preprocessing. The second experiment increases coupling
count relative to the baseline, with width restored to 64. The third combines
both changes. These are proposed comparisons, not measured improvements.

```sh
uv run --locked python -m flows.train --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --hidden-channels 96 --num-scales 2 --couplings-per-scale 3 --seed 42 --output checkpoints/realnvp_mnist_w96_c3_e30
```

```sh
uv run --locked python -m flows.train --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --hidden-channels 64 --num-scales 2 --couplings-per-scale 4 --seed 42 --output checkpoints/realnvp_mnist_w64_c4_e30
```

```sh
uv run --locked python -m flows.train --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --hidden-channels 96 --num-scales 2 --couplings-per-scale 4 --seed 42 --output checkpoints/realnvp_mnist_w96_c4_e30
```

Each command starts a fresh run and requires a new output directory. Compare
best validation BPD, fixed-seed generated samples, elapsed time, and the
training/validation curves. Use validation for selection; reserve test-set
evaluation for the selected configuration. Larger models may also overfit,
so compare their selected checkpoints rather than assuming epoch 30 is best.

## Architecture and direction

The forward direction maps data to Gaussian latents. A binary mask m keeps
coordinates with value one. The conditioner sees only m*x and predicts raw
scale and translation. On the complementary coordinates:

```text
s = 2*tanh(raw_scale)             # configurable bound; default 2
y = x*exp(s) + t
x = (y-t)*exp(-s)                # same kept coordinates give the same s,t
forward log|det J| = sum(s)      # inverse contributes -sum(s)
```

The conditioner is Conv3x3 -> ReLU -> Conv1x1 -> ReLU -> Conv3x3, with width 64.
This is the default `--conditioner-depth 3`; see the deeper-conditioner
experiment below for configurable additional convolutions.
The final convolution starts at zero, making each coupling initially identity.
Earlier conditioner layers initially have zero gradient until the final layer
moves away from zero. No batch normalization, dropout, or stochastic conditioner
layers are used. The bound limits per-layer exponential scales without altering
invertibility. There are no additional optimization tricks or objectives.

Each scale has three alternating checkerboard couplings, squeeze, then three
alternating channel couplings. Squeeze is a permutation with zero log absolute
determinant. After each nonfinal scale, the second half of channels is factored
out; the first half continues to the next scale.

```text
(1,28,28) -> (4,14,14) -> factor (2,14,14): 392 latents
                         active (2,14,14)
                         -> (8,7,7):       392 latents
                                           total: 784
```

Latents concatenate factored blocks in forward order followed by the final
active block. Inversion reverses scale order, reattaches the saved channels,
and reverses the channel/squeeze/checkerboard operations. Supported spatial
dimensions must be divisible by `2**num_scales`; MNIST supports two scales.
`AffineCoupling` also accepts vector masks and custom conditioners so the
transformation can be reused by a future latent-space VAE flow.

## Deeper coupling conditioners

`--conditioner-depth` counts all convolutions inside each coupling network,
including its input and output convolutions. It must be an integer of at least
3. Extra layers are hidden 3x3 convolutions followed by ReLU, inserted before
the output convolution. The input/output shapes and number of coupling
transformations stay the same.

```text
depth 3: Conv3x3 -> ReLU -> Conv1x1 -> ReLU -> Conv3x3
depth 5: Conv3x3 -> ReLU -> Conv1x1 -> ReLU
         -> Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> Conv3x3
```

At each operating resolution, the theoretical receptive field grows from 5x5
to 9x9 for depth 5. This experiment uses ordinary convolutions; it introduces
no residual blocks, normalization, scheduler, or objective changes. Deeper
conditioners do not themselves need to be invertible: the unchanged masked
coordinates still determine the same scale and translation in both directions.
The output convolution remains zero-initialized, so each coupling starts at
identity for any supported depth.

The first proposed depth comparison keeps the selected Fashion-MNIST setup:
128 hidden channels, eight couplings per mask family per scale, two scales,
batch size 64, scale limit 1, Adam learning rate 0.001, seed 42, and 30 epochs.
Only conditioner depth changes from 3 to 5. There are still 32 coupling
transformations, with 160 convolutions across their conditioners instead of 96.
The extra spatial convolutions increase parameter count, compute, and memory.
For this width-128, 32-coupling configuration, the instantiated models have
947,440 trainable parameters at depth 3 and 10,392,816 at depth 5.
Their effect on actual training speed and generation quality has not yet been
measured.

Run this experiment yourself:

```sh
uv run --locked python -m flows.train --dataset fashion_mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --hidden-channels 128 --num-scales 2 --couplings-per-scale 8 --conditioner-depth 5 --scale-limit 1.0 --logit-alpha 0.00001 --seed 42 --output checkpoints/realnvp_fashion_w128_c8_b64_scale1_d5_e30
```

Compare against `checkpoints/realnvp_fashion_w128_c8_b64_scale1_e30`, whose
selected epoch is 27 and validation BPD is 2.9514. Use validation and fixed-seed
sample grids for this comparison; no deeper-conditioner training results are
available yet.

New configurations and checkpoints record `conditioner_depth` under both
arguments and model settings. Existing checkpoints without that field load
the original depth-3 layout, with the same parameter names. The default remains
3, so previous training commands preserve the original architecture.

## Preprocessing and likelihood conventions

The shared loader returns q/255 for integer pixels q in {0,...,255}. Recover
q by rounding, then uniformly dequantize:

```text
u = (q + Uniform[0,1)) / 256
v = alpha + (1-2*alpha)*u           # alpha = 1e-5 by default
y = log(v) - log(1-v)              # optional; --no-logit makes y=u
z, flow_log_det = flow(y)
preprocess_log_det = sum(log(1-2*alpha) - log(v) - log(1-v))
log_prob = sum(log Normal(z;0,1)) + flow_log_det + preprocess_log_det - 784*log(256)
NLL = mean(-log_prob)              # nats per image; not per pixel
BPD = -log_prob / (784*log(2))     # then mean over images
```

The Gaussian normalization constant and every Jacobian are included. Factor-out
and permutations do not add a determinant term. The discrete bin correction is
applied exactly once. There is no image normalization, binarization, or padding
in addition to the operations above. Arbitrary continuous input images would be
quantized by the pixel-recovery operation; this API is for normalized discrete data.

Each image uses one uniform noise draw. Expected log_prob is the uniform
dequantization lower bound on log bin mass, so the reported NLL/BPD is an
upper-bound estimate, not an exact discrete likelihood. Contracted-logit
preprocessing maps the data cube to a finite logit interval. The Gaussian flow
can assign mass outside that interval (or outside the unit cube with no logit).
No truncation or renormalization for valid pixel support is performed. Sampling
inverts the exact transformations and can return values outside [0,1]; only the
saved display grids clip them. Likelihood computations do not clip samples.

## Training and artifacts

Adam minimizes mean per-image NLL, with learning rate 0.001 and batch size 64.
Defaults use a seeded random 90/10 train/validation split of the official
training set. Test data is never used for selection. Validation order and its
dequantization seed stay fixed each epoch; training noise changes each epoch.
This reduces validation noise when selecting checkpoints. Metrics weight all
images equally, including the last partial batch. There is no scheduler, mixed
precision, or gradient clipping in this baseline.

Each fresh run saves:

- `config.json`: architecture, preprocessing, seed, optimizer settings, split
  counts, runtime device, and PyTorch/CUDA details.
- `history.json`: epoch, train/validation NLL and BPD, and elapsed seconds.
- `best.pt`: model and optimizer states, configuration, selected epoch, and its
  validation metrics. Selection uses validation NLL only.
- `losses.png`: train/validation BPD curves.
- `samples_epoch_NNN.png`: fixed-seed sample grids every five epochs and at the end.
- `samples_best.png`: fixed-seed grid from the selected checkpoint.
- `summary.json`: best epoch, validation metrics, and test metrics if requested.

`--evaluate-test` reloads the best checkpoint and evaluates the official test
set once after training. Checkpoints are saved for selection and inspection;
exact training resumption, including all RNG states, is not implemented.
The run directory is ignored under `checkpoints/`. Copy important figures and
small metric/configuration files into the homework artifacts directory when
preparing the report; keep large weights and downloaded datasets ignored.

## Inspect a checkpoint

The experiment notebook loads results without starting training. In Python:

```python
from pathlib import Path
from flows.train import load_density_checkpoint
from shared import get_device

model, checkpoint = load_density_checkpoint(
    Path("checkpoints/realnvp_mnist_baseline/best.pt"), get_device()
)
images = model.sample(64)  # continuous samples; display range is [0,1]
```

## Correctness tests

```sh
uv run --locked pytest tests/test_flows.py tests/test_flow_training.py
```

The tests use only synthetic tensors and mocked dataset/training orchestration.
They cover complementary masks, finite-difference coupling Jacobians, nontrivial
forward/inverse reconstruction, squeeze permutations, factor-out/concatenation
ordering, complete multi-scale Jacobians, preprocessing Jacobians, Gaussian and
discrete-likelihood constants, finite densities, gradients, one synthetic
optimizer step, sample-weighted evaluation, checkpoint round trips, and selection
by validation metrics. A separate CUDA optimizer/inverse test runs if CUDA exists.
No test downloads MNIST or runs a dataset training experiment.
