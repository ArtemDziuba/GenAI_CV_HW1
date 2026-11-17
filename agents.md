
# AGENTS.md — Generative Models Homework

## 1. Project Context

This repository contains three university assignments on generative deep learning:

1. Normalizing Flows — RealNVP with coupling layers and multi-scale architecture.
2. Autoencoders — AE, VAE, flow-based VAE, and optionally beta-VAE.
3. Generative Adversarial Networks — vanilla GAN, conditional GAN, PatchGAN, feature matching, and optionally progressive GAN.

The full assignment descriptions are in `README.md`.

Read the relevant assignment requirements before planning or implementing changes. Treat the README as the source of truth for required deliverables.

The objective is to implement, train, evaluate, and understand these models, not merely to produce working code.

## 2. Agent Responsibilities

Act as a PyTorch research and implementation assistant.

- Prioritize mathematical correctness, clarity, and reproducibility.
- Implement only the functionality requested in the current task.
- Do not attempt to complete an entire homework unless explicitly requested.
- Prefer incremental changes that can be individually tested.
- Explain important architectural and mathematical choices.
- Prefer understandable implementations over unnecessary abstractions.
- Do not silently change model objectives or experiment definitions.
- Do not introduce additional architectures or optimization tricks without explaining their purpose.
- If a decision is ambiguous, choose a reasonable conventional approach and document the assumption. Ask when the choice would materially change the experiment.
- Preserve existing work unless modification is necessary.
- Do not use an external implementation as a substitute for implementing the assignment's core models.

## 3. Repository Organization

Use the following structure:

- `src/shared/` — dataset loading, training utilities, visualization, metrics, reproducibility.
- `src/flows/` — RealNVP and normalizing-flow components.
- `src/vae/` — AE, VAE, and flow-based VAE components.
- `src/gan/` — GAN generators, discriminators, and variants.
- `homeworks/01_normalizing_flows/` — Flow experiments and report.
- `homeworks/02_vae/` — AE/VAE experiments and report.
- `homeworks/03_gan/` — GAN experiments and report.
- `tests/` — model correctness and integration tests.
- `data/` — downloaded datasets.
- `checkpoints/` — saved model weights.

Keep model implementations in Python modules under `src/`.

Use Jupyter notebooks primarily for running experiments, visualizations, comparisons, and analysis.

Avoid duplicating shared components across homeworks.

In particular, the flow-based VAE should reuse appropriate RealNVP components from `src/flows/`.

## 4. Development Environment

- Python with PyTorch.
- Use `uv` for dependency management.
- Prefer existing project dependencies.
- Use CUDA when available, otherwise CPU.
- Do not hardcode device names or machine-specific absolute paths.
- Use `pathlib` for filesystem paths.
- Use torchvision for downloading MNIST, Fashion-MNIST, and CIFAR-10 where required.
- Use matplotlib for visualizations.
- Use pytest for automated tests.

Do not add heavy dependencies or experiment-tracking frameworks unless necessary.

Prefer solutions that can run on a consumer GPU with approximately 8 GB VRAM.

## 5. Implementation Standards

### General

- Use modular PyTorch `nn.Module` implementations.
- Use clear and descriptive function and variable names.
- Include type hints for public functions where practical.
- Document non-obvious mathematical operations and tensor shapes.
- Avoid unnecessary inheritance and abstraction layers.
- Avoid hardcoded architecture dimensions when configuration is reasonable.
- Keep training loops separate from model definitions.
- Prefer straightforward, idiomatic PyTorch.

### Mathematical Correctness

Correctness is more important than training speed.

- Follow the mathematical definitions of the models.
- Clearly distinguish training objectives from evaluation metrics.
- Document loss reductions and normalization conventions.
- Verify tensor dimensions and broadcasting.
- Avoid numerical instability in probability and logarithm calculations.
- Do not silently change signs, scales, or constants in likelihood calculations.
- Do not substitute a different objective simply because it is easier to implement.
- State assumptions when results depend on preprocessing or probability distributions.

### Model-Specific Requirements

Normalizing Flows:
- Ensure coupling transformations are invertible.
- Verify forward/inverse consistency.
- Verify log-determinant calculations.
- Handle image preprocessing and dequantization consistently.
- Use correct change-of-variables likelihood calculations.

AE / VAE:
- Separate encoder, decoder, and latent sampling logic.
- Implement reparameterization correctly.
- Distinguish reconstruction loss, KL divergence, and total objective.
- Make latent dimensionality configurable.
- Handle the transformed posterior density correctly for flow-based VAEs.

GAN:
- Separate generator and discriminator updates.
- Correctly manage gradient flow and parameter updates.
- Make class conditioning explicit in conditional GANs.
- Monitor generator/discriminator losses and generated samples.
- Implement and evaluate architectural modifications separately.

## 6. Experiments and Reproducibility

- Use explicit random seeds for Python, NumPy, and PyTorch.
- Do not claim exact reproducibility when nondeterministic operations are used.
- Separate training, validation, and test evaluation.
- Do not use test metrics for selecting hyperparameters.
- Save relevant hyperparameters with experimental results.
- Log training and validation metrics where applicable.
- Use appropriate model-specific evaluation metrics.
- Keep generated sample grids comparable across experiments.
- For GAN comparisons, use fixed latent noise and labels where applicable.
- Do not fabricate experimental results, metrics, or conclusions.
- Clearly distinguish actual measurements from expected behavior.

Use small smoke tests before launching expensive training runs.

Do not start long-running training or download large datasets without explicit authorization.

## 7. Testing and Verification

Add tests for significant model components.

At minimum, verify:
- Expected input and output tensor shapes.
- Successful forward passes.
- Finite losses and outputs.
- Correct gradient propagation.
- Small optimization steps without runtime errors.

Additional checks:

Flows:
- Forward/inverse reconstruction.
- Log-determinant consistency.
- Invertibility across supported dimensions.

VAE:
- Reparameterization behavior.
- Correct output shapes.
- KL divergence calculation.
- Finite ELBO and gradients.

GAN:
- Generator output shapes.
- Discriminator output shapes.
- Successful independent optimization steps.
- Conditional label handling.

Run relevant tests after modifying an implementation.

If a test cannot run, explain why. Never report tests as passing unless they were executed.

## 8. Notebooks and Reports

Each homework should contain one or more experiment notebooks and a Markdown report.

Notebooks should:
- Be organized into clearly labeled sections.
- Import models from `src/` rather than duplicate them.
- Show relevant configurations and metrics.
- Save important plots and generated sample grids.
- Avoid excessive logs or unnecessary outputs.

Reports should address:
1. Implementation and experimental setup.
2. Results and required artifacts.
3. What worked well.
4. What did not work as expected.
5. Possible explanations for limitations.
6. Potential improvements.

Do not invent interpretations or experimental findings.

The student is responsible for final scientific conclusions. The agent may help organize, explain, and critique the results.

## 9. Scope and Change Management

Before substantial implementation:
1. Inspect the relevant existing code.
2. Read the corresponding README requirements.
3. Briefly outline the proposed changes.
4. Identify reusable components.
5. Implement the requested changes.
6. Run relevant tests.
7. Summarize changes and verification results.

Avoid unrelated refactoring.

Do not:
- Rewrite unrelated code.
- Delete experiment results without permission.
- Change dependency versions without justification.
- Overwrite trained checkpoints.
- Introduce unnecessary infrastructure.
- Commit generated datasets, large checkpoints, or temporary files.
- Create Git commits unless explicitly requested.

## 10. Completion Criteria

A task is complete when:
- The requested functionality is implemented.
- Relevant correctness checks pass.
- Code is readable and consistent with repository conventions.
- Important design decisions are documented.
- No unrelated functionality was modified.

At the end of each task, provide:
- Summary of implemented changes.
- Files created or modified.
- Tests executed and their outcomes.
- Any remaining limitations or recommended next steps.
