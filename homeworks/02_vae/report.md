# Autoencoders and VAE report

Status: scaffold only. No model, training run, or measured results yet.
Requirements: [assignment README](../../readme.md).

## Implementation and experimental setup

TODO: Describe AE, VAE, and flow-based VAE implementations under `src/vae/`.
Reuse appropriate RealNVP components from `src/flows/` for the latent flow.
Record dataset (MNIST and Fashion-MNIST), transforms, splits, seeds, device,
latent sizes, hyperparameters, and checkpoints. Define the reconstruction
distribution, loss reductions, KL term, ELBO, and transformed posterior density.

## Results and required artifacts

- [ ] Compare AE compression and reconstructions at several latent sizes.
- [ ] Plot reconstruction error against latent size and save reconstruction grids.
- [ ] Train a VAE on MNIST and Fashion-MNIST and compare results.
- [ ] Plot reconstruction, KL, and total objective histories with stated reductions.
- [ ] Report final test metrics separately from validation selection metrics.
- [ ] Save VAE samples drawn from the latent prior.
- [ ] Train a flow-based VAE and save metrics and generated samples.
- [ ] Optional: explore beta-VAE weights, disentanglement, reconstruction,
      generation diversity, and latent traversals.
- [ ] Link figures, measured metrics, configurations, and checkpoints.

TODO: Add actual measurements and artifact links. No results are available yet.

## What worked well

TODO: Discuss observations supported by the experiments.

## What did not work as expected

TODO: Describe observed failures or limitations.

## Possible explanations for limitations

TODO: Separate evidence from hypotheses.

## Potential improvements

TODO: Suggest follow-up experiments based on the measured results.
