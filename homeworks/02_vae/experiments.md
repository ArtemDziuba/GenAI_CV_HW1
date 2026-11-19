# Task 2: training and generation experiments

## Saved AE compression analysis

The [AE compression figures and source metrics](artifacts/ae_compression_01/compression.md)
use each saved run's best-checkpoint validation MSE. They include dataset
overviews, the best available result per latent size, and one plot per matching
backbone/training configuration. Only measured latent sizes are shown. A group
with one latent size requires additional training for a curve; no such runs are
launched by this analysis command.

To export a new snapshot to a fresh directory, without loading models or datasets:

```powershell
uv run --locked python -m vae.compare --model ae --runs-root checkpoints/vae --output homeworks/02_vae/artifacts/ae_compression_next
```

Run commands from the repository root in PowerShell. Training is launched by you;
no dataset training was performed while implementing this task. README is unchanged.
Use a fresh output directory for every command. Dependencies already include what
this implementation needs; `uv sync --locked` installs the project.

## Implemented models

- **AE:** deterministic convolutional encoder, configurable vector bottleneck,
  mirrored decoder with sigmoid output. It compresses/reconstructs images. There
  is no fitted prior, so random Gaussian AE latents are not a valid generation baseline.
- **VAE:** same image backbone, separate Gaussian mean/log-variance heads,
  reparameterized posterior samples, standard normal prior. Log variance is
  explicitly bounded to [-12,12] for numerical stability.
- **f-VAE:** the Gaussian posterior sample passes through alternating affine
  couplings from `flows.AffineCoupling`, using fully connected conditioners.
  Conditioners share weights across images; input dependence comes through the
  Gaussian posterior and its sample. This flow changes the posterior, not the
  prior: generation still draws `z ~ N(0,I)` and calls the decoder directly.
- **Optional beta-VAE:** `--beta` scales KL during optimization. The ordinary
  negative ELBO is still logged separately and used for checkpoint selection.
  Prior-coordinate traversal grids support qualitative exploration of individual
  dimensions; they do not by themselves demonstrate disentanglement.

The encoder downsamples 28 -> 14 -> 7 with 4x4 stride-2 convolutions of widths
`C, 2C`, then a dense feature layer. The decoder reverses this using transposed
convolutions. `--conv-depth 1` means one convolution per resolution stage;
larger values add 3x3 stride-1 layers. Activations are SiLU. There is no batch
normalization, dropout, scheduler, KL warm-up, or learned observation variance.

## Observations, objectives, and comparisons

Both datasets use the existing seeded loader: **54,000 train / 6,000 validation /
10,000 test**, with seed 42 and validation fraction 0.1. Labels are ignored;
these are unconditional models. The split matches Task 1 for the same seed.
Subset flags intentionally reduce these counts for smoke tests.

AE uses original grayscale pixels and by default minimizes **SSE/image**, then averages over
the batch. `grayscale_mse` is SSE/784, averaged over images. A latent size L
corresponds to 784/L input scalars per latent scalar; this is a dimension ratio,
not a compressed-file byte ratio or an entropy-coded bitrate.
`--ae-loss l1` instead minimizes summed absolute error per image. Both per-pixel
MSE and MAE are logged for AE runs. Checkpoint selection stays on validation
MSE for these controlled comparisons. MSE and L1 objective magnitudes have
different meanings and should not be directly ranked against each other.

VAE/f-VAE offer two explicitly different observation protocols:

| Option | Encoder input and target | Reconstruction likelihood |
| --- | --- | --- |
| `--likelihood bernoulli` (default) | Dynamically binarized pixels, `x_binary ~ Bernoulli(x_gray)` | Exact Bernoulli log probability on that binary image, evaluated stably using logits |
| `--likelihood gaussian --observation-std 0.1` | Original grayscale pixels | Normal density with sigmoid mean and fixed standard deviation; includes the normalization constant |

Dynamic binarization changes every training epoch. Evaluation uses a fixed local
seed for binarization and posterior sampling. Keep evaluation seed, batch size,
and image ordering fixed for comparisons: the Monte Carlo draws depend on them.
Gaussian observations retain Fashion-MNIST shading, so this is a useful *separate*
experiment family. The Gaussian has support on all real pixel values; only
display grids clip noisy draws to [0,1].

For the VAE, `KL = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)` is analytic.
For the f-VAE:

```text
z0 ~ q0(z|x), zK = f(z0), log_det = log|det(dzK/dz0)|
log qK(zK|x) = log q0(z0|x) - log_det
KL estimate = log qK(zK|x) - log N(zK;0,I)
objective = reconstruction_nll + beta * KL
negative_elbo = reconstruction_nll + KL
```

Every likelihood/KL term sums over pixels/latents, then takes the batch mean:
**nats/image**. A sampled flow KL can be negative for individual observations;
it is not clamped. The negative ELBO is an estimated upper bound on marginal
NLL in expectation, not exact NLL. Beta != 1 changes the training objective;
the beta objective is not labeled as the ordinary ELBO.

`vae.evaluate` also estimates a negative importance-weighted bound using K
posterior draws per fixed observed image. This remains an estimate, not an
exact marginal likelihood. Gaussian continuous densities can exceed one and
their NLL can be negative. Neither protocol is the dequantized 8-bit BPD from
Task 1: **do not compare those numbers directly**, or pool Bernoulli and Gaussian
ELBO values into one ranking. Changing Gaussian std changes the objective too.

Checkpoints minimize validation negative ELBO (AE: grayscale MSE), never test
metrics. For beta sweeps, this makes the default checkpoint criterion explicit;
inspect saved epoch grids for your reconstruction/diversity preferences as well.
Test evaluation is an explicit final action after choosing configurations.

## First commands

Show all settings:

```powershell
uv run --locked python -m vae.train --help
```

First run a small f-VAE smoke experiment on the GPU:

```powershell
uv run --locked python -m vae.train --model fvae --dataset mnist --device cuda --epochs 3 --train-limit 1024 --validation-limit 256 --latent-dim 16 --hidden-channels 32 --hidden-dim 256 --conv-depth 1 --flow-steps 4 --flow-hidden-dim 128 --scale-limit 1 --batch-size 64 --output checkpoints/vae/smoke_fvae_mnist
```

The datasets are already present. Add `--download` only if needed. `--device auto`
allows CPU fallback; explicit CUDA raises a clear error if unavailable.

Run all three baseline models on **both** datasets, sequentially, for 30 epochs:

```powershell
foreach ($dataset in @("mnist", "fashion_mnist")) {
    foreach ($model in @("ae", "vae", "fvae")) {
        uv run --locked python -m vae.train --model $model --dataset $dataset --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --latent-dim 16 --hidden-channels 32 --hidden-dim 256 --conv-depth 1 --likelihood bernoulli --beta 1 --flow-steps 4 --flow-hidden-dim 128 --flow-hidden-layers 2 --scale-limit 1 --seed 42 --output "checkpoints/vae/${model}_${dataset}_z16_c32_d1_base_e30"
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $model / $dataset" }
    }
}
```

Flow flags apply only to f-VAE; likelihood/beta apply only to variational models.
AE always reconstructs grayscale. The ordinary VAE always has zero flow steps.

## Tunable hyperparameters

Suggested values below are candidate experiments, not measured recommendations.
Change one variable at a time initially. Defaults target a practical starting
point, rather than assuming the best RealNVP settings transfer to a VAE.

| CLI parameter | Default | Candidate values | What it controls |
| --- | --- | --- | --- |
| `--model` | `vae` | `ae`, `vae`, `fvae` | Compression only vs Gaussian posterior vs posterior flow |
| `--latent-dim` | 16 | 2, 4, 8, 16, 32, 64 | Compression capacity and prior-space dimension; f-VAE requires >=2 |
| `--hidden-channels` | 32 | 16, 32, 64, 96 | Image encoder/decoder width (second stage is twice this) |
| `--hidden-dim` | 256 | 128, 256, 512 | Dense image feature and decoder projection width |
| `--conv-depth` | 1 | 1, 2, 3 | Convolutions per stage; extra spatial processing in encoder and decoder |
| `--beta` | 1 | 0.25, 0.5, 1, 2, 4 | Reconstruction versus posterior-to-prior regularization; larger can reduce information or cause collapse, smaller can worsen prior samples despite good reconstructions |
| `--flow-steps` | 4 (f-VAE) | 2, 4, 8 | Posterior expressiveness, cost, and Jacobian accumulation; plain VAE is the zero-flow comparison |
| `--flow-hidden-dim` | 128 | 64, 128, 256 | Width of each coupling conditioner MLP |
| `--flow-hidden-layers` | 2 | 1, 2, 3 | Hidden linear layers per coupling MLP, excluding output |
| `--scale-limit` | 1 | 0.5, 1, 2 | Bounds each posterior coupling log-scale; does not change observation noise |
| `--likelihood` | `bernoulli` | `bernoulli`, `gaussian` | Binary-image probability versus grayscale continuous density |
| `--observation-std` | 0.1 | 0.05, 0.1, 0.2 | Gaussian observation noise; changes reconstruction/KL balance and likelihood constants |
| `--learning-rate` | 0.001 | 0.0001, 0.0003, 0.001 | Adam update magnitude |
| `--batch-size` | 64 | 32, 64, 128 | Gradient noise, memory use, updates per epoch; record wall time too |
| `--epochs` | 30 | Start at 30; extend selected runs | Optimization budget; best validation checkpoint may be earlier |
| `--weight-decay` | 0 | 0, 0.00001, 0.0001 | Optional Adam weight penalty; baseline has none |
| `--grad-clip` | 0 (off) | 0, 5, 10 | Optional global gradient-norm clipping; does not change the written objective |
| `--seed` | 42 | 42, 123, 2026 | Repeats; also changes the train/validation split, so pair seeds across models |
| Evaluation `--temperature` | 1 | 0.7, 0.85, 1, 1.15 | Prior sampling spread at inference only; T!=1 changes the sampled latent distribution |
| Evaluation `--importance-samples` | 32 | 1, 32, 128 | Likelihood-bound estimation budget, not a training or visual-quality parameter |

Keep `--validation-fraction 0.1` fixed for comparisons. `--num-workers` is a
throughput setting, not model capacity. `--sample-count` and `--sample-every`
control artifacts. `--train-limit`/`--validation-limit` are for smoke tests;
omit them from final runs. `--deterministic` requests deterministic PyTorch
algorithms; seeds alone do not guarantee bitwise reproducibility on all devices.

## Experiment order and commands

1. **AE compression requirement:** run the latent sweep below, keeping other
   settings fixed, and compare validation MSE and reconstructed details.
2. **VAE baseline:** compare the two datasets at z16, beta1. Inspect generated
   decoder means AND actual observation draws; monitor reconstruction NLL and KL.
3. **Generation sweeps:** first latent size 8/16/32/64, then beta 0.5/1/2/4
   around the preferred size. Consider width 64 or conv-depth 2 separately.
4. **f-VAE:** compare 2/4/8 posterior couplings using the same image backbone,
   observation protocol, latent size, beta, and budget as VAE. A richer posterior
   need not automatically improve prior-sample sharpness.
5. **Grayscale likelihood:** compare a separate Gaussian family on both datasets
   if binarization loses useful shading, especially on Fashion-MNIST.
6. Repeat selected configurations with multiple paired seeds, then evaluate their
   held-out test sets. Do not choose configurations from test scores.

AE latent sweep on both datasets:

```powershell
foreach ($dataset in @("mnist", "fashion_mnist")) {
    foreach ($latent in @(2, 4, 8, 16, 32, 64)) {
        uv run --locked python -m vae.train --model ae --dataset $dataset --device cuda --epochs 30 --batch-size 64 --latent-dim $latent --hidden-channels 32 --hidden-dim 256 --conv-depth 1 --seed 42 --output "checkpoints/vae/sweep_ae_${dataset}_z${latent}_e30"
        if ($LASTEXITCODE -ne 0) { throw "AE sweep failed: $dataset / $latent" }
    }
}
```

VAE beta sweep (after the baseline; output names are distinct):

```powershell
foreach ($dataset in @("mnist", "fashion_mnist")) {
    foreach ($beta in @(0.5, 1, 2, 4)) {
        uv run --locked python -m vae.train --model vae --dataset $dataset --device cuda --epochs 30 --batch-size 64 --latent-dim 16 --hidden-channels 32 --hidden-dim 256 --conv-depth 1 --likelihood bernoulli --beta $beta --seed 42 --output "checkpoints/vae/sweep_vae_${dataset}_z16_beta${beta}_e30"
        if ($LASTEXITCODE -ne 0) { throw "Beta sweep failed: $dataset / $beta" }
    }
}
```

Gaussian comparison, preserving grayscale on both datasets:

```powershell
foreach ($dataset in @("mnist", "fashion_mnist")) {
    uv run --locked python -m vae.train --model vae --dataset $dataset --device cuda --epochs 30 --batch-size 64 --latent-dim 16 --likelihood gaussian --observation-std 0.1 --beta 1 --seed 42 --output "checkpoints/vae/vae_${dataset}_z16_gaussian_std01_e30"
    if ($LASTEXITCODE -ne 0) { throw "Gaussian run failed: $dataset" }
}
```

Export a CSV/JSON comparison and AE error-versus-latent-size figure:

```powershell
uv run --locked python -m vae.compare --runs-root checkpoints/vae --output homeworks/02_vae/artifacts/comparison_01
```

The scatter plot shows each completed AE run separately, including its dataset.
Use only the controlled sweep for conclusions about latent size; the CSV retains
capacity settings and seeds so unrelated runs are not silently averaged.

Evaluate a selected checkpoint on validation with 32 importance samples and
generate a lower-temperature display experiment (metrics still use T=1):

```powershell
uv run --locked python -m vae.evaluate --checkpoint checkpoints/vae/vae_mnist_z16_c32_d1_base_e30/best.pt --device cuda --split validation --batch-size 64 --importance-samples 32 --temperature 0.85 --output homeworks/02_vae/artifacts/mnist_vae_validation_t085
```

After choosing final settings, explicitly evaluate the official test set for both
datasets; substitute the actual selected run paths:

```powershell
foreach ($dataset in @("mnist", "fashion_mnist")) {
    uv run --locked python -m vae.evaluate --checkpoint "checkpoints/vae/vae_${dataset}_z16_c32_d1_base_e30/best.pt" --device cuda --split test --batch-size 64 --importance-samples 128 --seed 43 --output "homeworks/02_vae/artifacts/final_vae_${dataset}_test"
    if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed: $dataset" }
}
```

## Saved artifacts and what to inspect

Each training run saves `config.json` (architecture, objective, split counts,
runtime and parameter count), `history.json`, `summary.json`, and `best.pt`
(weights, optimizer, configuration, epoch, validation metrics). Runs cannot
overwrite previous experiments. Epoch grids refer to that epoch's model;
`*_best.png` grids refer to the selected checkpoint. Training resume is not
implemented; a checkpoint supports inference/evaluation, not exact RNG replay.

AE saves objective/MSE curves and reconstruction grids. VAE/f-VAE additionally
save reconstruction-NLL, KL, negative-ELBO curves, prior decoder-mean grids,
actual observation draws, and latent traversals. Reconstruction grids use the
transformed **base posterior mean** for a stable display; this is not generally
the mean of the flow posterior and differs from sampled training reconstructions.
Latent random seeds are fixed across epoch sample grids and comparable runs
with the same latent dimension. Image display always uses [0,1].

Reconstruction previews now default to **24 validation images in a 3x8 grid**.
Each comparison cell places its original above its reconstruction (with a binary
input between them for Bernoulli VAEs). A companion `*_only.png` contains just
the reconstructed images. Set `--reconstruction-count` and
`--reconstruction-columns` in either `vae.train` or `vae.evaluate` to change
the layout; fewer images are used if the selected split/subset is smaller.
This changes visualization only and does not require retraining old checkpoints.

For example, regenerate the smaller MNIST AE's epoch-30 checkpoint preview:

```powershell
uv run --locked python -m vae.evaluate --checkpoint checkpoints/vae/ae_mnist_z16_c16_h64_d1_e30/best.pt --device cuda --split validation --reconstruction-count 24 --reconstruction-columns 8 --output checkpoints/vae/ae_mnist_z16_c16_h64_d1_e30/preview_24
```

The checkpoint epoch is recorded in `evaluation.json`. Existing run images and
checkpoint weights are preserved; use a fresh output directory for each export.

For generation, compare recognizability, diversity, repeated modes, shading,
and reconstructions alongside validation bounds. Very small KL with poor
reconstructions can indicate posterior collapse; good reconstructions with
poor prior samples can indicate a posterior/prior mismatch. These are hypotheses
to check against actual outputs, not findings from unrun experiments.

## MNIST AE smoothing experiments: separate commands

Each experiment starts from the smaller baseline (C16, H64, Z16, depth 1),
changing one setting. Keep 30 epochs, batch 64, Adam lr 0.001, and seed 42.
These commands train fresh models; none resumes an earlier checkpoint. The
same validation split and 24 preview images are used. Run experiments individually.

1. **More convolutions: depth 1 -> 2.**

```powershell
uv run --locked python -m vae.train --model ae --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --latent-dim 16 --hidden-channels 16 --hidden-dim 64 --conv-depth 2 --ae-loss mse --reconstruction-count 24 --reconstruction-columns 8 --seed 42 --output checkpoints/vae/ae_mnist_z16_c16_h64_d2_e30
```

2. **Wider dense layers: hidden dimension 64 -> 128.**

```powershell
uv run --locked python -m vae.train --model ae --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --latent-dim 16 --hidden-channels 16 --hidden-dim 128 --conv-depth 1 --ae-loss mse --reconstruction-count 24 --reconstruction-columns 8 --seed 42 --output checkpoints/vae/ae_mnist_z16_c16_h128_d1_e30
```

3. **Larger latent code: latent dimension 16 -> 32.** This also changes the
   scalar dimension ratio from 49:1 to 24.5:1, so it is a compression trade-off.

```powershell
uv run --locked python -m vae.train --model ae --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --latent-dim 32 --hidden-channels 16 --hidden-dim 64 --conv-depth 1 --ae-loss mse --reconstruction-count 24 --reconstruction-columns 8 --seed 42 --output checkpoints/vae/ae_mnist_z32_c16_h64_d1_e30
```

4. **L1 reconstruction objective:** keep the smaller baseline architecture,
   replace summed squared error with summed absolute error. This is an objective
   experiment; sharper edges are a hypothesis, not a guaranteed outcome.

```powershell
uv run --locked python -m vae.train --model ae --dataset mnist --device cuda --epochs 30 --batch-size 64 --learning-rate 0.001 --latent-dim 16 --hidden-channels 16 --hidden-dim 64 --conv-depth 1 --ae-loss l1 --reconstruction-count 24 --reconstruction-columns 8 --seed 42 --output checkpoints/vae/ae_mnist_z16_c16_h64_d1_l1_e30
```

Compare validation MSE, validation MAE where available, and stroke/loop details
in the same 24 reconstructions. Old AE checkpoints without an `ae_loss` field
remain evaluable and are interpreted as MSE models. New run exports include
the AE loss choice in configuration and comparison tables. Dataset training
for these commands is launched by the student.
