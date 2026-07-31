# Stage 5: Diffusion Model Extension (updated)

The diffusion model, the upgrade from the Stage 4 VAE baseline. This is the
**updated v2 model**: trained on 8,040 cells, conditioned on **volume fraction
and stiffness class**, with EMA weights. It is the model exported to ONNX and
deployed in the browser app. It supersedes the original vf-only model trained on
480 cells.

## Contents
- `ddpm2.py` — model, training (resumable, chunked), and sampling.
- `generate2.py` — the final pipeline: calibrated request -> diffusion ->
  rejection by the evaluator (vf tolerance + periodic connectivity).
- `metrics.py` — the Stage 3 evaluator (used by `generate2.py` for the filter).
- `checkpoints/ddpm2_ep25.pt` — selected checkpoint (EMA weights + stiffness bin edges).
- `model/ddpm2_ep25.onnx` — browser export (8 MB, opset 17).
- `model/EXPORT_NOTES.md` — ONNX interface, calibration map, measured controllability.
- `samples/` — stiffness rows, a volume-fraction sweep, and pipeline output.
- `README.md` — this file.

## Model
- Small U-Net noise predictor (~2.0M params), 32×32, sinusoidal time embedding.
- DDPM with 400 noise steps; sampled with **DDIM** (40–50 steps) for speed.
- **Classifier-free guidance**: the condition is dropped 12% of the time in
  training; at sampling, guidance `w` (default 3) pushes toward the target.
- **EMA** of the weights; samples come from the EMA copy.

### Conditioning (4-dim)
`cond = [volume_fraction, stiff_low, stiff_med, stiff_high]` — a continuous volume
fraction plus a one-hot stiffness class (terciles of the homogenized modulus).
Labels come from `unitcells_big/labels_cond.csv` (volume fraction + modulus).

## Running it
```
pip install numpy scipy torch pillow

# train (resumable; repeat to continue). ~150-720 s/epoch on CPU depending on cores.
python ddpm2.py train --epochs 2

# sample directly from the model
python ddpm2.py sample --vf 0.45 --stiff high -n 16 --guidance 3.0

# final pipeline: calibrated request + rejection filter (recommended)
python generate2.py --vf 0.5 --stiff high -n 16 --symmetrize
```

## Results (epoch 25, the deployed checkpoint)
- **Checkpoint chosen by controllability, not loss.** Epoch 29 had lower loss but
  weaker conditioning; epoch 25 was selected by a vf-tracking + stiffness-
  separation probe.
- **Volume fraction:** monotonic and calibrated. Requests 0.2/0.4/0.6 give
  achieved ~0.37/0.44/0.51. Reliable target range ~**0.37–0.57** (the response is
  monotonic but compressed). The calibration map is in `model/EXPORT_NOTES.md` and
  in `generate2.py`.
- **Stiffness:** low vs high separates by about **2.4×** in effective modulus at a
  fixed volume fraction (the capability the Stage 4 model and the original
  vf-only diffusion model lacked). The medium class is the weakest link.
- **Connectivity:** 22–24 of 24 samples periodically connected.
- **Rejection yield:** ~40% at vf 0.40 and ~72% at vf 0.50 (up from 8% and 25%
  for the original 480-cell model — the larger dataset's main payoff).

## Browser export
`model/ddpm2_ep25.onnx` exports the EMA weights with a wrapper that returns both
the conditional and unconditional noise (`eps_c`, `eps_u`) so the browser does
classifier-free guidance in JavaScript. Verified: per-step parity vs PyTorch
2e-6, and the full sampling loop matches the PyTorch sampler at **0.9989 IoU**.
Interface and the JS scheduler details are in `model/EXPORT_NOTES.md`.

## Honest limits
- **Volume-fraction range is compressed** to ~0.37–0.57; requests outside saturate.
  Widening it needs a stronger conditioning signal, not more epochs (more training
  past epoch 25 weakened control).
- **Specklier than the VAEs predecessor:** the 32×32 organic outputs carry stray
  isolated pixels, so component counts are higher even when the main structure
  percolates. The rejection filter guarantees a connected load path but does not
  remove the speckle.
- **No auxetic conditioning.** The training data has essentially no negative-
  Poisson cells, so auxetic cannot be targeted; `generate2.py` cannot request it.
- Trained on synthetic data; the architecture scales unchanged to real
  topology-optimized cells, which is the main planned upgrade.

## Relationship to the original (v1)
The original Stage 5 model (`ddpm.py`, checkpoint `ddpm_ep108`, 480 cells, vf-only
conditioning) is superseded by this one. The original report's comparison was
re-run against this v2 model; see the updated Stage 6 report.
