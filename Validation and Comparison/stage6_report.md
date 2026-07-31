# Do Generative Models Produce Better Metamaterial Candidates Than Random Generation? (Stage 6, updated)

*Updated validation. This compares four design sources against the **deployed v2
diffusion model** (trained on 8,040 cells, conditioned on volume fraction and
stiffness, EMA weights) rather than the original 480-cell vf-only model. Every
cell is scored by the same Stage 3 evaluator (periodic homogenization, plane
stress, solid E = 1, 8-connectivity) at 32x32, so the numbers are comparable to
the original report. Reproduce with `compare.py`.*

## Setup

Four sources, 16 cells per group per target, at two target volume fractions:

- **random** — Gaussian-smoothed noise thresholded to the exact target vf (so zero
  vf error by construction).
- **manual** — parametric cells from the dataset whose measured vf is near the target.
- **vae** — the Stage 4 conditional VAE, conditioned on the target (vf, high
  stiffness, biaxial symmetry), mirror-symmetrized, then evaluated at 32x32.
- **diffusion** — the Stage 5 v2 model: calibrated request + rejection by the
  evaluator (vf tolerance and periodic connectivity).

Targets are **0.40 and 0.50** (not the original 0.30 and 0.50): the v2 model's
reliable volume-fraction range is ~0.37-0.57, so 0.30 is below its floor.

## Results

The `achieved vf` column is included because it is needed to read the stiffness
column honestly: a denser cell is trivially stiffer.

Target volume fraction 0.40:

| group | n | achieved vf | vf error | components | connected | symmetry | E_eff |
|---|---|---|---|---|---|---|---|
| random | 16 | 0.400 | 0.000 | 7.6 | 0.00 | 0.594 | 0.0008 |
| manual | 16 | 0.397 | 0.021 | 1.1 | 0.50 | 1.000 | 0.0519 |
| vae | 16 | 0.525 | 0.125 | 4.8 | 0.62 | 1.000 | 0.1473 |
| diffusion | 16 | 0.397 | 0.019 | 10.1 | 1.00 | 0.667 | 0.0477 |

Target volume fraction 0.50:

| group | n | achieved vf | vf error | components | connected | symmetry | E_eff |
|---|---|---|---|---|---|---|---|
| random | 16 | 0.500 | 0.000 | 5.4 | 0.44 | 0.578 | 0.0126 |
| manual | 16 | 0.518 | 0.024 | 1.4 | 0.94 | 0.989 | 0.2744 |
| vae | 16 | 0.610 | 0.138 | 1.8 | 0.81 | 1.000 | 0.2465 |
| diffusion | 16 | 0.504 | 0.013 | 5.0 | 1.00 | 0.651 | 0.0954 |

Diffusion rejection yield: **40%** at vf 0.40 (19 of 48) and **66%** at vf 0.50
(21 of 32). The original 480-cell model managed 8% and 25%.

## Interpretation

**Generative beats random, decisively.** Random patterns win only the vf column,
trivially, because the threshold is set to the target. Structurally they are
unusable: at vf 0.40 not one random cell forms a connected load path and their
stiffness is essentially zero (0.0008). The diffusion pipeline at the same material
budget delivers 100% connected cells with real stiffness.

**Diffusion is the most steerable source.** It hits the requested volume fraction
within 0.01-0.02 and is 100% connected after rejection, at both targets. That
controllability, hitting a property target on demand, is the actual product of the
project; the manual family cannot vary continuously and the VAE misses the target.

**The VAE's stiffness is inflated by overshoot, not a real win.** It looks stiff
(0.147 and 0.246), but it overshoots the requested volume fraction by 0.11-0.14:
asked for 0.40 it produces 0.525, asked for 0.50 it produces 0.610. A denser cell
is stiffer for free, so this is a failure to hit the target, not a stiffness
advantage. Held at matched density the point is clear: at vf ~0.52 the VAE gives
0.147 while manual gives ~0.27. Judged on the target, the VAE is the **least
steerable generator** (its vf error is 6-7x the diffusion model's). The Stage 4
blur-then-threshold behavior is the cause: the VAE settles toward dense, smooth
fills that symmetrize cleanly but ignore the requested budget.

**Manual designs still win on clean quality.** They keep ~1 component, near-perfect
symmetry, and the highest stiffness at matched density (0.274 at vf 0.50). As in
the original report, this is expected: the manual group is the clean training
distribution itself. The generators' value is not beating a known design at its own
game but producing diverse candidates at a requested property.

**Two honest weaknesses of the deployed diffusion model.** It is **specklier** than
the manual and VAE cells (5-10 components: the main structure percolates, but stray
pixels remain), and its **symmetry is lower** (~0.65) because it was not trained
with a symmetry constraint. Both are cosmetic for stiffness but matter for
manufacturability, and both have known fixes (symmetry folding at sampling,
guidance that penalizes floating material).

## Conclusion

The answer remains **yes, with qualifications**, and the deployed model strengthens
it. Generative models produce candidates categorically better than random:
connected where random fragments, several times stiffer at the same material
budget, and steerable to a requested volume fraction within 0.02 plus a stiffness
class. Among the generators, **diffusion is clearly the better tool**: it hits the
target, while the VAE only appears competitive by overshooting density. Neither
matches clean manual designs in absolute stiffness or symmetry, the expected
outcome for models trained on synthetic data, and the diffusion model is specklier
than ideal. The remaining weaknesses point to the same upgrades: training on real
topology-optimized data, and guidance-based sampling that steers toward connected,
symmetric designs instead of discarding failures afterward.

## Reproducibility

`compare.py` regenerates everything below. Per-cell scores are in
`comparison_all_cells.csv` (128 rows: 4 groups x 2 targets x 16 cells). Figures
`fig_compare_vf0.4.png` and `fig_compare_vf0.5.png` show eight cells per group,
rows ordered random, manual, vae, diffusion. Components: `metrics.py` (evaluator),
`cvae.py` + `checkpoints/cvae.pt` (VAE), `ddpm2.py` + `checkpoints/ddpm2_ep25.pt`
(diffusion), dataset `unitcells_big/`.
