# Metamaterial Unit-Cell Datasets and Generation

Synthetic binary unit cells used to train the generative models in this project,
plus the scripts that produce and label them. Pixel value 255 (white) = solid
material, 0 (black) = void. Each image is one periodic unit cell; tiling it makes
the metamaterial.

---

## The datasets

### training_dataset_8040/   — the dataset the deployed model was trained on
- **images/** — 8,040 PNGs, 64×64, across six families (holes, lattice, cross,
  ring, bars, re-entrant). The diffusion model loads them at 32×32.
- **labels.csv** — columns: `filename, volume_fraction, E_mean_rel`
  - `volume_fraction` — measured fraction of solid pixels.
  - `E_mean_rel` — effective Young's modulus from periodic homogenization,
    relative to the solid material (solid E = 1), averaged over x and y.
  - Stiffness classes used for conditioning are terciles of `E_mean_rel`:
    **low < 0.0194, medium < 0.1088, high ≥ 0.1088**.

### original_dataset_480/    — the smaller set used for the VAE and first diffusion model
- **images/** — 480 PNGs, 64×64 (80 per family).
- **labels.csv** — the full evaluator output (volume fraction, connectivity,
  symmetry, manufacturability, full homogenized elastic properties).

### Honest note on the data and labels
`volume_fraction` and `E_mean_rel` are **measured, not assumed**. An earlier
version carried by-construction "auxetic" labels; those were removed after the
homogenizer showed the synthetic re-entrant cells are not actually auxetic.

These are **synthetic** cells, a working stand-in for real topology-optimization
output. They span a useful range of volume fraction and stiffness but contain
essentially no auxetic (negative-Poisson) cells, which is why the models cannot
target auxetic behavior. Swapping in real topology-optimized data is the main
planned upgrade; every downstream script runs unchanged on it.

---

## Regenerating from scratch

Two steps: generate the geometry (images), then label it.

### Requirements
```
pip install numpy scipy pillow
```

### 1. Generate the images — `generate_unitcells.py`

Produces the binary 64×64 cells across the six families and a basic `labels.csv`
(volume fraction, connectivity, symmetry, manufacturability, approximate
stiffness).

Edit the config block at the top of the file:
```
SIZE = 64                 # image side length
N_PER_CATEGORY = 80       # cells per family  (80 -> 480 total, 1340 -> 8040 total)
SEED = 42                 # reproducibility
OUT_DIR = "unitcells"     # output folder
```

Reproduce the two datasets above:
```
# the 480-cell set (VAE + first diffusion model)
#   set N_PER_CATEGORY = 80,  OUT_DIR = "unitcells"
python generate_unitcells.py

# the 8,040-cell set (the deployed diffusion model)
#   set N_PER_CATEGORY = 1340, OUT_DIR = "unitcells_big"
python generate_unitcells.py
```
Each run writes `OUT_DIR/images/*.png` and `OUT_DIR/labels.csv`. Generation is
fast (the 8,040-cell set takes a few seconds).

**How the geometry is made:** each family is a parametric generator (random
circles for holes, periodic bars for lattices, struts for trusses, an annulus for
rings, inward-angled walls for re-entrant). About half the cells are folded to
enforce mirror symmetry. Cells with degenerate volume fraction (near-empty or
near-solid) are rejected and regenerated. See the family functions in
`generate_unitcells.py`.

### 2. Label for conditioning — `label_dataset.py`

The diffusion model conditions on volume fraction + a stiffness class, so each
cell needs a measured modulus. This script adds that via fast conjugate-gradient
homogenization at 24×24, writing `<folder>/labels_cond.csv` (this is the
`labels.csv` shipped in `training_dataset_8040/`).
```
python label_dataset.py unitcells_big 1500   # run repeatedly (resumable, in chunks)
```
Columns: `filename, volume_fraction, E_mean_rel`. About 6 minutes for 8,040 cells
on a CPU. Stiffness terciles are computed from the `E_mean_rel` column at
training time.

### 3. (Optional) Full evaluation — `metrics.py`

The richer evaluator: exact volume fraction, periodic connectivity, symmetry,
manufacturability, and the full homogenized elastic tensor (E, bulk K, shear G,
Poisson). This produced the labels in `original_dataset_480/`, and it scores
generated designs.
```
python metrics.py unitcells/images --out unitcells/metrics.csv   # whole folder
python metrics.py cell.png                                       # one cell, report
```
