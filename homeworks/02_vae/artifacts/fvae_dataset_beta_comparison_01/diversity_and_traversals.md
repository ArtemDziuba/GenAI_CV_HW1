# Generation diversity and latent-coordinate inspection

This qualitative review uses existing results only. No training, new model inference,
dataset downloads, or checkpoint changes were performed.

## Scope and comparison protocol

The representative models are f-VAEs with Z32 C32 H256 D1 F4 FH64, trained with
beta 1 or 4 on each dataset. All use Bernoulli observations with dynamic binarization,
30 training epochs, batch 64, Adam learning rate 0.001 and training seed 42.
The selected checkpoints are epoch 30 for both Digits runs and Fashion beta1,
and epoch 29 for Fashion beta4. Checkpoint selection uses validation negative ELBO.

For diversity, the inspected figures are the six saved comparison grids below:
64 prior decoder means per model at temperature 1, with sampling seeds 42, 123,
and 2026. Thus each representative model has 192 displayed prior draws. The
F4/FH64 model is the middle panel of each row; beta1 is above beta4.

- Digits: [seed 42](mnist_seed_42.png), [seed 123](mnist_seed_123.png), [seed 2026](mnist_seed_2026.png).
- Fashion: [seed 42](fashion_mnist_seed_42.png), [seed 123](fashion_mnist_seed_123.png), [seed 2026](fashion_mnist_seed_2026.png).

The sampling seeds are inference repeats, not independent training repeats.
The same latent vector need not produce the same class in different trained
models. These displays show decoder means, without added observation noise.

## What generation diversity means

Diversity has at least two visually distinguishable parts:

1. **Between-category variety:** different digit identities, or different clothing
   and footwear silhouettes, rather than one repeated category.
2. **Within-category variety:** different handwriting styles, widths and slants,
   or different garment outlines and appearances within a similar category.

Recognizability and sharpness are separate properties. A set of sharp images can
still repeat one style, while noisy or malformed images can differ strongly in
pixels without providing useful semantic variety.

| Dataset | beta1 observations | beta4 observations | Supported interpretation |
| --- | --- | --- | --- |
| Digits | Multiple recognizable digit identities and visibly varied stroke widths, loops and slants; some ambiguous shapes. | Multiple digit identities remain visible, with generally softer, simpler stroke shapes. | Both show category variety. beta1 appears to preserve more fine handwriting variation; beta4 does not visibly improve diversity consistently. |
| Fashion | A mixture of tops, trousers, dresses, bags and shoe-like silhouettes, with visible differences in brightness and outline. Some samples are ambiguous. | Multiple silhouette families remain present, but many garments are smoother and more generic, with weaker internal detail. | Broad category variety remains at both weights. beta1 appears richer in appearance variation; reduced detail at beta4 should not be mistaken for disappearance of all categories. |

Neither setting visually collapses to one repeated output in the inspected random
grids. This does not establish balanced class frequencies, complete coverage of
the real distribution, or absence of memorization. No classifier labels,
class-frequency statistics, or numerical diversity scores were computed.

## How to read the latent traversals

Each saved traversal has eight rows and nine columns:

- Row 1 varies the first latent coordinate, row 2 the second, through row 8.
- Values run left to right through -3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3.
- All other 31 coordinates remain zero; the image is decoded directly from that vector.
- Every row's middle image is therefore the same decode of the all-zero vector.
  This repetition is expected and is not evidence of generation collapse.

For f-VAE these are **prior-coordinate** traversals, not traversals through the
posterior flow. The flow changes posterior inference during training; generation
uses the standard Gaussian prior and decoder directly.

A candidate separated factor would be a row that changes, for example, only
stroke thickness while retaining digit identity, size and position. Smooth image
changes alone do not demonstrate such separation. A nearly unchanged row only
shows weak sensitivity along that particular direction around the zero anchor.

## Digits: observed coordinate effects

### beta1, selected epoch 30

![Digits beta1 traversal](../../../../checkpoints/vae/fvae_mnist_z32_c32_h256_d1_f4_fh64_beta1_e30/traversal_best.png)

Counting rows from the top, row 3 changes a broad rounded, zero-like form through
an eight-like form into a narrow, one-like form. Width, loop structure and apparent
identity change together. Row 7 likewise moves from angular shapes through the
central eight-like image toward nine-like images, mixing style and identity.
Several other rows change loop proportions or stroke geometry, while row 4 has
relatively little visible effect.

The directions have recognizable visual effects, but these examples do not isolate
independent controls for identity, thickness, slant or width.

### beta4, selected epoch 30

![Digits beta4 traversal](../../../../checkpoints/vae/fvae_mnist_z32_c32_h256_d1_f4_fh64_beta4_e30/traversal_best.png)

Rows 3, 4, 5 and 8 remain visually similar over much of the displayed range.
Rows 1, 2, 6 and 7 change digit-like structure more visibly, including loop
topology and apparent identity. These active directions still mix visual changes.

Compared with beta1, more of these first eight directions are nearly unchanged
around the chosen anchor. That is consistent with more concentrated use of
coordinates, but is not a measurement of posterior activity or proof that the
remaining coordinates are unused. Clearer independent semantic controls are not
established by this figure.

## Fashion: observed coordinate effects

### beta1, selected epoch 30

![Fashion beta1 traversal](../../../../checkpoints/vae/fvae_fashion_mnist_z32_c32_h256_d1_f4_fh64_beta1_e30/traversal_best.png)

Row 1 changes a bright upright garment-like shape through darker clothing forms
toward a boot-like silhouette. Brightness, shape and apparent category change
together. Row 3 changes garment structure and eventually produces wider, lower
forms. Row 7 changes the brightness and outline of top/coat-like images.
Rows 2, 4, 5, 6 and 8 show comparatively small changes at this anchor.

There are meaningful directions, but no inspected row provides a reliable
brightness-only, garment-length-only or category-only control.

### beta4, selected epoch 29

![Fashion beta4 traversal](../../../../checkpoints/vae/fvae_fashion_mnist_z32_c32_h256_d1_f4_fh64_beta4_e30/traversal_best.png)

Rows 1 and 7 show the most conspicuous changes, mixing brightness and silhouette
and eventually reaching low, horizontal forms. Most other displayed rows remain
close to the central coat-like image. This is evidence of limited directional
sensitivity in this particular view, rather than a demonstration of separated
semantic factors.

## Conclusion and limits

The inspected models generate multiple recognizable categories and appearance
variations. Increasing beta from 1 to 4 generally simplifies or smooths the
generated appearances in these grids. It does not provide convincing visual
evidence of better-separated semantic controls in the inspected traversals.

Meaningful factors appear **partially represented but still entangled**: a single
coordinate can change several properties together. This is a qualitative finding,
not a quantitative disentanglement score or a claim about every coordinate.

Only the first eight of 32 coordinates were inspected, all around one zero anchor.
Coordinate meanings can differ between separately trained models. Flat rows can
be locally insensitive yet behave differently at other anchors; active rows may
also change meaning elsewhere. Stronger follow-up evidence would require all
coordinates, multiple prior or encoded-image anchors, and checking whether the
same property changes consistently across examples. Such additional inference
has not been performed for this review. Additional training is not needed merely
to extend that inspection using the existing checkpoints.
