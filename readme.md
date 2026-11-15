For each homework, please include (could be in repo) a brief report that covers the following:
- All necessary artefacts: plots of loss/metrics, results of image generation, etc.
- A short discussion of your results and findings:
    - What worked well?
    - What didn’t turn out as expected?
    - Why do you think the results might not be great?
    - How could the results be improved?

The report doesn't need to be too formal — just clearly explain your process, insights, and reflections.
Note: You are encouraged to use AI tools to assist with any part of this work — including code autocompletion, explanations, or drafting parts of the report. You may also refer to external implementations, libraries, or repositories as guidance or inspiration. Just ensure that the core ideas, reasoning, and final deliverables are your own work.

## Norm Flow 5pt.
Implement RealNVP model and train it on mnist datasets. 

- Build the RealNVP architecture (you could extend from lecture example) for grayscale MNIST Digits and MNIST Fashion images
- Use Coupling Layer and Multi-Scale
- Implement train loop with validation metrics, evaluate on test set
- Train the model until it achieves reasonable likelihood and visualize generated samples by inverting the flow.
- Experiments with hyperparameters or architecture to get nice results of generation. You should try to get somewhat reasonable generation quality for generation. 

## VAE - 10pt. (+2pt.)

Autoencoder Compression:
- Implement a simple autoencoder for the MNIST datasets.
- Train it to compress and reconstruct input images.
- Experiment with different latent (hidden) sizes to analyze their effect on reconstruction quality and compression error.
- Plot the reconstruction error vs. latent size and visualize sample reconstructions for comparison

Variational Autoencoder (VAE):
- Implement a VAE for both MNIST Digits and Fashion-MNIST datasets.
- Train the model to achieve reasonable reconstruction quality and likelihood.
- Visualize generated samples by sampling from the latent space.
- Compare results between MNIST Digits and Fashion-MNIST.

f-VAE:
- Use VAE and RealNVP to implement NF inside VAE latent space
- Train the model to achieve reasonable reconstruction quality and likelihood.
- Visualize generated samples by sampling from the latent space.

Bonus (optional): β-VAE Exploration (2pt.)
- Implement a β-VAE variant by adjusting the KL divergence weight (β).
- Explore how varying β affects:
    - Latent space disentanglement
    - Reconstruction quality
    - Generation diversity
- Visualize the effect of changing individual latent dimensions on generated images.

## GAN - 10pt. (+2pt.)
Vanilla GAN
- Implement a basic GAN pipeline consisting of a generator and discriminator.
- Train the GAN on MNIST and Fashion-MNIST datasets.
- Monitor training stability by plotting generator and discriminator losses over time.
- Evaluate the quality of generated samples and check for mode collapse by visualizing generated batches during training.

Conditional and Advanced GAN Variants
- Implement a Class-Conditional GAN (cGAN) where the generator is conditioned on class labels.
- Extend the discriminator by experimenting with: PatchGAN architecture, Feature Matching Loss
- Compare the visual quality and diversity of generated samples before and after these modifications.

Bonus (optional): Progressive Growing GAN - 2pt.
- Implement a Progressive GAN and train it on the CIFAR-10 dataset.
- Gradually increase image resolution during training to improve stability and generation quality.
- Visualize the evolution of generated images as the model grows.