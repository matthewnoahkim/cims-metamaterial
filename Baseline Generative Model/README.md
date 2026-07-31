# Stage 4: Baseline Generative Model (Conditional VAE)

The baseline generator, built before the diffusion model: a convolutional
variational autoencoder (VAE) that reconstructs unit cells and generates new ones,
with a conditional version steered by target properties.

## Contents
- `cvae.py` — model, training, and sampling (both the unconditional VAE and the
  conditional CVAE).
- `train_chunk.py` — resumable chunked trainer (keeps each run short on CPU).
- `checkpoints/vae.pt` — trained unconditional VAE.
- `checkpoints/cvae.pt` — trained conditional VAE (conditions on volume fraction,
  stiffness class, symmetry class).
- `samples/` — example outputs (reconstructions, unconditional and conditional
  samples, and a training-vs-generated comparison).
- `README.md` — this file.

## Model
- Convolutional encoder/decoder, latent dimension 32, ~1.07M parameters, 64×64.
- Encoder: four stride-2 conv layers (32→64→128→128), then linear heads for the
  latent mean and log-variance. Decoder mirrors it with transposed convolutions.
- Loss: binary cross-entropy reconstruction + KL divergence, with a KL warmup
  over the first third of training so the model learns to reconstruct before the
  latent is regularized.

### Conditioning (the CVAE)
The condition is a 9-dimensional vector:
- volume fraction (1 continuous value),
- stiffness class (one-hot, 3: low / medium / high, from terciles of the
  homogenized modulus),
- symmetry class (one-hot, 5: square / biaxial / uniaxial / diagonal / none).

It is concatenated to the encoder input as constant image channels and to the
latent code at the decoder, the standard CVAE recipe. Labels come from the Stage 3
evaluator (`unitcells/metrics.csv`), the same quantities used to score outputs.

## Running it
```
pip install numpy torch pillow

# train both models (writes checkpoints/vae.pt and checkpoints/cvae.pt)
python cvae.py train --epochs 150

# or train one model in resumable chunks (repeat to continue)
python train_chunk.py vae 50
python train_chunk.py cvae 50

# generate conditional samples
python cvae.py sample --vf 0.3 --stiff low --sym biaxial -n 16 --symmetrize
```
Importable: `load(name)`, `sample(model, n, cond, symmetrize=...)`,
`make_cond(vf, stiff, sym)`.

## Results (trained on the 480-cell dataset)
- **Reconstruction IoU 0.984** — the autoencoder faithfully encodes/decodes cells.
- **Volume-fraction conditioning works**, with some compression: requests
  0.3 / 0.5 / 0.6 produce ~0.22 / 0.41 / 0.61.
- **Stiffness conditioning** lands the requested tercile in roughly 70–100% of
  samples depending on the target.
- **Symmetry conditioning through the label alone failed** (0 of 16), so symmetry
  is enforced at sampling time by mirror-averaging the decoder's probability field
  before thresholding (the `--symmetrize` flag). With it, symmetry is exact.
- **Novelty check:** nearest-training-neighbour IoU averages ~0.68 (max ~0.90),
  so the model interpolates rather than memorizing — worth re-checking after any
  retrain on a small dataset.

## Honest findings
- The VAE is the **weaker generator** in this project. Its volume-fraction error
  grows with the target and its samples **fragment at low density**: the
  blur-then-threshold behavior of pixel VAEs breaks thin struts when the smooth
  decoder output is binarized. This is the measured motivation for moving to the
  diffusion model (Stage 5), which produces cleaner, more connected structures.
- Trained on only 480 synthetic cells, so treat the numbers as a pipeline
  demonstration, not a performance claim. The architecture scales unchanged to
  larger data.
- The `samples/` figures show this directly: clean reconstructions and recognizable
  conditional structure, but speckle and fragmentation at low volume fraction.

## Files this depends on
- Dataset images at `unitcells/images/` and labels at `unitcells/metrics.csv`
  (the Stage 3 output). Adjust paths at the top of `cvae.py` if your layout differs.
