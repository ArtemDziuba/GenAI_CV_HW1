# GAN report

Status: scaffold only. No model, training run, or measured results yet.
Requirements: [assignment README](../../readme.md).

## Implementation and experimental setup

TODO: Describe the vanilla GAN, class conditioning, PatchGAN, and feature
matching experiments. Put generators and discriminators in `src/gan/`.
Record MNIST/Fashion-MNIST transforms, splits, seeds, device, optimizers,
hyperparameters, and checkpoints. Define objectives and describe separate
generator/discriminator updates. Keep latent noise and class labels fixed for
comparable sample grids. Evaluate each modification separately.

## Results and required artifacts

- [ ] Train a vanilla GAN on MNIST and Fashion-MNIST.
- [ ] Save generator and discriminator loss curves and generated batches over time.
- [ ] Discuss observed training stability, sample quality, diversity, and mode collapse.
- [ ] Train a class-conditional GAN and show samples with explicit class labels.
- [ ] Evaluate PatchGAN and feature matching separately and compare with the baseline.
- [ ] Optional: implement progressive growing on CIFAR-10 and show generated
      images as resolution grows. CIFAR-10 loading is outside this scaffold.
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
