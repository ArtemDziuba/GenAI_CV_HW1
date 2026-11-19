# Task 2 final test evaluation

Both requested steps are complete. Six frozen checkpoints were evaluated once each on the full official 10,000-image test split of their dataset. No training, downloads, or original checkpoint changes occurred.

## Selection and protocol

Step 1 selects the lowest saved full-split validation MSE for AE, and the lowest saved Bernoulli validation ordinary negative ELBO for VAE and f-VAE. Test metrics were not used to select checkpoints. The variational selections all have beta=1.

Step 2 matches each model family between Digits and Fashion-MNIST. The six step 1 winners already form three exact architecture/objective pairs, so this step reuses those evaluations rather than resampling or testing additional models.

Evaluation uses CUDA, batch 64, seed 42, original grayscale AE targets, and the recorded Bernoulli binarization protocol for VAE/f-VAE. Variational results include a 32-sample importance-weighted bound. Likelihood terms are nats/image and MSE is per pixel. Negative ELBO/IWAE are estimated bounds, not exact marginal NLL. Their separate passes use seeded stochastic observation/posterior draws. Finite-sample estimates need not obey a strict sample-wise ordering.

## Step 1: MNIST Digits

| Model | Z/C/H/D | Epoch | Val MSE | Test MSE | Val negative ELBO | Test negative ELBO | Test negative IWAE-32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 64/32/256/1 | 29 | 0.001765 | 0.001733 | n/a | n/a | n/a |
| VAE | 32/64/256/1 | 30 | 0.010893 | 0.010818 | 91.2263 | 90.6115 | 86.1672 |
| FVAE | 32/32/256/1; F4/FH64 | 30 | 0.010810 | 0.010791 | 90.5943 | 90.0439 | 85.8296 |

## Step 1: Fashion-MNIST

| Model | Z/C/H/D | Epoch | Val MSE | Test MSE | Val negative ELBO | Test negative ELBO | Test negative IWAE-32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 64/32/256/1 | 28 | 0.004881 | 0.004841 | n/a | n/a | n/a |
| VAE | 32/64/256/1 | 30 | 0.014198 | 0.014148 | 232.7867 | 233.4992 | 230.3982 |
| FVAE | 32/32/256/1; F4/FH64 | 30 | 0.014498 | 0.014463 | 232.7821 | 233.5651 | 230.7178 |

## Step 2: matched architectures between datasets

Step 1 already supplies exact cross-dataset architecture matches, including the same likelihood, beta and flow settings. These rows reuse the same six evaluations.

| Model | Identical Z/C/H/D | Parameters per model | Digits test MSE | Fashion test MSE | Digits test negative ELBO | Fashion test negative ELBO |
| --- | --- | --- | --- | --- | --- | --- |
| AE | 64/32/256/1 | 1,708,801 | 0.001733 | 0.004841 | n/a | n/a |
| VAE | 32/64/256/1 | 3,507,137 | 0.010818 | 0.014148 | 90.6115 | 233.4992 |
| FVAE | 32/32/256/1; F4/FH64 | 1,742,337 | 0.010791 | 0.014463 | 90.0439 | 233.5651 |

## Observations

Test minus validation negative-ELBO gaps (nats/image) are mnist VAE: -0.6148; mnist FVAE: -0.5504; fashion_mnist VAE: +0.7125; fashion_mnist FVAE: +0.7831. The AE test MSEs are also close to their validation values. On Digits the selected f-VAE has a lower test bound than the selected VAE; on Fashion the selected VAE is slightly lower. The image backbones differ between these model families, and only one seed is evaluated, so this does not establish a general flow advantage. Step 2 matches backbones across datasets within each family, not across AE/VAE/f-VAE.

## Interpretation limits

These are one training seed and one evaluation seed per configuration. Small differences are not statistical rankings. AE uses grayscale inputs and deterministic reconstruction; the variational models use binarized inputs and sampled posterior reconstructions. Their MSE differences therefore do not isolate architecture. These metrics do not directly measure prior generation quality, diversity or disentanglement. No additional tuning follows from test scores.

## Individual artifacts

- mnist AE, checkpoint epoch 29: [metrics](step1/mnist/ae/evaluation.json), [24 test reconstructions](step1/mnist/ae/reconstructions.png).
- fashion_mnist AE, checkpoint epoch 28: [metrics](step1/fashion_mnist/ae/evaluation.json), [24 test reconstructions](step1/fashion_mnist/ae/reconstructions.png).
- mnist VAE, checkpoint epoch 30: [metrics](step1/mnist/vae/evaluation.json), [24 test reconstructions](step1/mnist/vae/reconstructions.png), [prior decoder means](step1/mnist/vae/samples.png), [observation draws](step1/mnist/vae/observations.png), [prior traversal](step1/mnist/vae/traversal.png).
- mnist FVAE, checkpoint epoch 30: [metrics](step1/mnist/fvae/evaluation.json), [24 test reconstructions](step1/mnist/fvae/reconstructions.png), [prior decoder means](step1/mnist/fvae/samples.png), [observation draws](step1/mnist/fvae/observations.png), [prior traversal](step1/mnist/fvae/traversal.png).
- fashion_mnist VAE, checkpoint epoch 30: [metrics](step1/fashion_mnist/vae/evaluation.json), [24 test reconstructions](step1/fashion_mnist/vae/reconstructions.png), [prior decoder means](step1/fashion_mnist/vae/samples.png), [observation draws](step1/fashion_mnist/vae/observations.png), [prior traversal](step1/fashion_mnist/vae/traversal.png).
- fashion_mnist FVAE, checkpoint epoch 30: [metrics](step1/fashion_mnist/fvae/evaluation.json), [24 test reconstructions](step1/fashion_mnist/fvae/reconstructions.png), [prior decoder means](step1/fashion_mnist/fvae/samples.png), [observation draws](step1/fashion_mnist/fvae/observations.png), [prior traversal](step1/fashion_mnist/fvae/traversal.png).

[Frozen selections, commands and hashes](manifest.json), [metrics CSV](metrics.csv), [metrics JSON](metrics.json). Per-model logs are saved alongside this file.
