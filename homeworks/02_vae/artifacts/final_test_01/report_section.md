### Final test evaluation

Both steps use the full official 10,000-image test split per dataset. Checkpoints were frozen using validation results before evaluation. CUDA, batch 64, seed 42; variational models use Bernoulli observations, beta=1 and an additional IWAE bound with 32 importance samples. Likelihood terms are nats/image; MSE is per pixel. No retraining was performed.

#### Step 1: top validation performers

**MNIST Digits**

| Model | Z/C/H/D | Epoch | Val MSE | Test MSE | Val negative ELBO | Test negative ELBO | Test negative IWAE-32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 64/32/256/1 | 29 | 0.001765 | 0.001733 | n/a | n/a | n/a |
| VAE | 32/64/256/1 | 30 | 0.010893 | 0.010818 | 91.2263 | 90.6115 | 86.1672 |
| FVAE | 32/32/256/1; F4/FH64 | 30 | 0.010810 | 0.010791 | 90.5943 | 90.0439 | 85.8296 |

**Fashion-MNIST**

| Model | Z/C/H/D | Epoch | Val MSE | Test MSE | Val negative ELBO | Test negative ELBO | Test negative IWAE-32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 64/32/256/1 | 28 | 0.004881 | 0.004841 | n/a | n/a | n/a |
| VAE | 32/64/256/1 | 30 | 0.014198 | 0.014148 | 232.7867 | 233.4992 | 230.3982 |
| FVAE | 32/32/256/1; F4/FH64 | 30 | 0.014498 | 0.014463 | 232.7821 | 233.5651 | 230.7178 |

#### Step 2: matched architectures between datasets

Step 1 already supplies exact cross-dataset architecture matches, including the same likelihood, beta and flow settings. These rows reuse the same six evaluations.

| Model | Identical Z/C/H/D | Parameters per model | Digits test MSE | Fashion test MSE | Digits test negative ELBO | Fashion test negative ELBO |
| --- | --- | --- | --- | --- | --- | --- |
| AE | 64/32/256/1 | 1,708,801 | 0.001733 | 0.004841 | n/a | n/a |
| VAE | 32/64/256/1 | 3,507,137 | 0.010818 | 0.014148 | 90.6115 | 233.4992 |
| FVAE | 32/32/256/1; F4/FH64 | 1,742,337 | 0.010791 | 0.014463 | 90.0439 | 233.5651 |

Negative ELBO and IWAE are likelihood-bound estimates. AE and variational MSE also differ in observation protocol and deterministic versus sampled reconstruction, so their values do not isolate architecture effects. These test results do not measure generation diversity or disentanglement.

[Full test results, figures and provenance](artifacts/final_test_01/results.md).

Test minus validation negative-ELBO gaps (nats/image) are mnist VAE: -0.6148; mnist FVAE: -0.5504; fashion_mnist VAE: +0.7125; fashion_mnist FVAE: +0.7831. The AE test MSEs are also close to their validation values. On Digits the selected f-VAE has a lower test bound than the selected VAE; on Fashion the selected VAE is slightly lower. The image backbones differ between these model families, and only one seed is evaluated, so this does not establish a general flow advantage. Step 2 matches backbones across datasets within each family, not across AE/VAE/f-VAE.
