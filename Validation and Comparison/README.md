# Stage 6: Validation and Comparison

Does a generative model produce better metamaterial candidates than random
generation? This stage answers it by comparing four design sources, scored by the
same Stage 3 evaluator, against the deployed v2 diffusion model.

## Contents
- `compare.py` — reproducible comparison: builds the four groups, scores every
  cell, writes the per-cell CSV, prints the summary table, saves the figures.
- `metrics.py` — the Stage 3 evaluator used for all scoring.
- `comparison_all_cells.csv` — per-cell scores (128 rows: 4 groups x 2 targets x 16).
- `stage6_report.md` — the write-up with results, interpretation, and conclusion.
- `fig_compare_vf0.4.png`, `fig_compare_vf0.5.png` — eight cells per group,
  rows ordered random, manual, vae, diffusion.
- `README.md` — this file.

## Running it
```
pip install numpy scipy torch pillow
python compare.py        # targets 0.40 and 0.50, 16 cells per group
```
Needs `cvae.py` + `checkpoints/cvae.pt` (Stage 4), `ddpm2.py` +
`checkpoints/ddpm2_ep25.pt` (Stage 5), and the dataset at `unitcells_big/`.

## The four sources
- **random** — Gaussian-smoothed noise thresholded to the exact target vf.
- **manual** — parametric dataset cells near the target.
- **vae** — the Stage 4 conditional VAE (conditioned, symmetrized).
- **diffusion** — the Stage 5 v2 model (calibrated request + rejection filter).

## CSV columns
`group, target_vf, volume_fraction, n_solid_components, connected,
symmetry_score, E_eff_x_rel, E_eff_y_rel, poisson_eff`. One row per cell.

## Headline finding
Generative beats random decisively (connected vs fragmented, stiff vs limp,
steerable to within 0.02 vf). Among the generators, **diffusion is the better
tool**: it hits the target volume fraction, while the VAE only looks competitive
on stiffness because it overshoots density by 0.11-0.14 (read the `achieved vf`
column in the report before the stiffness column). Clean manual designs still win
on absolute stiffness and symmetry. Full reasoning in `stage6_report.md`.

## Honest notes
- Targets are 0.40 and 0.50, not 0.30, because the v2 model's reliable vf range is
  ~0.37-0.57.
- The diffusion cells are specklier (5-10 components) and less symmetric (~0.65)
  than manual/VAE cells; rejection guarantees a connected load path but not the
  removal of stray pixels.
- Numbers vary slightly run to run (stochastic sampling); rerun `compare.py` to
  regenerate.
