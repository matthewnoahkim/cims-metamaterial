# Stage 3: Simple Mechanical Analysis

The evaluator that scores a unit-cell image on design quality and mechanical
behavior. It labels the dataset and grades generated designs with the same code,
so a generated cell is judged on measured performance, not pixel similarity.

## Contents
- `metrics.py` — the evaluator (design-quality metrics + periodic homogenization).
- `stage3_metrics_480.csv` — `metrics.py` over the 480-cell dataset (one row per cell).
- `stage3_metrics_8040.csv` — the same full 27-column analysis over the 8,040-cell
  training set (the dataset the deployed model was trained on).
- `README.md` — this file.

## Running it
```
pip install numpy scipy pillow

python metrics.py cell.png                       # one cell, readable report
python metrics.py cell.png --json                # one cell, raw dict (incl. CH tensor)
python metrics.py images/ --out stage3_metrics.csv   # whole folder -> CSV
python metrics.py cell.png --no-fem              # skip homogenization (fast, geometry only)
python metrics.py cell.png --min-feature 4       # set min printable width in pixels
python metrics.py cell.png --target '{"volume_fraction":0.5,"E_eff_x_rel":0.4}'  # performance error vs a target
```
Importable too: `compute_metrics(array)`, `property_vector(array)` (the
conditioning/target vector), and `evaluate_against_target(metrics, target)`.

## Convention
A pixel is **solid** when its value is > 127 (white = solid, black = void).
Stiffness is **relative to the solid material (solid E = 1)**; multiply by your
real Young's modulus for physical units.

## CSV columns

Geometry and connectivity
- `volume_fraction` — fraction of solid pixels.
- `symmetry_class` — square / biaxial / uniaxial / diagonal / none.
- `symmetry_score` — 0–1, best mirror match (1.0 = perfectly symmetric).
- `mirror_x`, `mirror_y`, `diagonal` — individual symmetry booleans.
- `n_solid_components` — number of disconnected solid pieces (8-connectivity; 1 ideal).
- `largest_component_fraction` — share of solid in the largest piece.
- `percolates_x`, `percolates_y` — solid spans edge to edge in x / y.
- `periodic_connected_x`, `periodic_connected_y` — solid joins its tiled copies
  (the correct connectivity test for a repeating unit cell).

Manufacturability
- `min_feature_px` — the minimum printable width used for the checks (default 3).
- `thin_solid_fraction` — fraction of solid in members thinner than that.
- `thin_void_fraction` — fraction of void in gaps narrower than that.
- `floating_solid_fraction` — solid not attached to the main structure.
- `trapped_void_fraction` — enclosed voids (bad for powder/casting processes).
- `est_min_solid_thickness_px` — estimated thinnest member, from a distance transform.
- `manufacturable` — true when thin / floating / trapped fractions are all small.

Effective mechanical properties (periodic homogenization, plane stress)
- `E_eff_x_rel`, `E_eff_y_rel` — effective Young's modulus in x and y.
- `poisson_eff` — effective Poisson's ratio.
- `G_eff_rel` — effective shear modulus.
- `K_eff_rel` — effective 2D (area) bulk modulus.
- `auxetic` — true when Poisson's ratio is negative (below −0.01).
- `specific_stiffness_rel` — stiffness-to-weight = mean effective modulus ÷ volume fraction.

## How the homogenization works
For each macroscopic unit strain (xx, yy, shear) it solves the periodic
equilibrium problem on a Q4 pixel mesh and integrates the energy to get the
effective stiffness tensor, from which E, ν, K, and G follow. The full evaluator
uses a direct solver at the native resolution; a faster conjugate-gradient
version at 24×24 is used elsewhere for bulk labeling and in the browser.

## Validation
The homogenizer is checked against known answers:
- a fully solid cell returns E = 1.000, ν = 0.300, G = 0.385 (the base material);
- a half-solid / half-void layered cell returns E = 0.50 in the stiff direction
  and ~0 in the soft direction (the Voigt and Reuss bounds);
- an auxetic base material (ν = −0.3) is recovered exactly, confirming the
  Poisson sign is reported correctly.

## Honest limits
- **Linear elastic, plane stress.** Covers stiffness, bulk/shear modulus, and
  auxetic detection, but not large-strain or energy-absorption behavior (those
  need a nonlinear solver).
- **Stiffness/Poisson approximations elsewhere.** The CSV here uses the full-
  resolution solver; the 24×24 CG version used for bulk labeling and the browser
  is a close approximation, not identical.
- **Manufacturability uses thresholds**, not a specific machine's process limits;
  adjust `min_feature_px` to your process.
- In this dataset only ~1 in 480 cells is auxetic, consistent with the project's
  finding that simple synthetic geometry is almost never auxetic.
- **Filter by connectivity before trusting stiffness or Poisson.** For
  disconnected or near-empty cells the effective tensor is near-singular, so its
  Poisson's ratio can be wildly out of the physical range (a real connected 2D
  material is bounded to roughly −1…1; the 8,040-set CSV contains a few values
  near ±6 from such cells). Keep rows where `periodic_connected_x` and
  `periodic_connected_y` are both true (about 5,400 of the 8,040) for meaningful
  mechanical numbers; the auxetic flag on a disconnected cell is not reliable.
