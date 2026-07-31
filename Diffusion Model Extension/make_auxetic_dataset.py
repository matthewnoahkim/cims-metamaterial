"""
Auxetic training cells for the nu-conditioned DDPM (v3).

Family: rotating squares with corner hinges -- four rigid squares per cell,
alternately rotated +/-alpha, tied by small hinge disks at the lattice-edge
midpoints. Hinge flexure gives negative effective Poisson's ratio. Validated
in the browser demo work (2026-07): nu down to about -0.65 at vf 0.40-0.55,
resolution-consistent between the native grid and a 2x-refined FE solve.

Cells are drawn NATIVE at 32x32 (the DDPM's resolution) and saved as 64x64
PNGs (0 void / 255 solid) so they pass through the existing training resize
(64 -> 32 bilinear + threshold) without any change to the drawn geometry.

Labels use the SAME Stage-3 evaluator (metrics.compute_metrics on the 64x64
image) as the 8040-cell dataset, so nu is measured, never assumed.

Usage:  python make_auxetic_dataset.py [--n 2000] [--seed 0]
"""
import os
import csv
import argparse
import hashlib
import numpy as np
from PIL import Image
from metrics import compute_metrics

M = 32                                   # native draw grid = DDPM resolution
OUT = r"C:\Users\matth\OneDrive\CIMS Lab\Dataset Preparation\auxetic_dataset"
IMG_DIR = os.path.join(OUT, "images")
CSV_PATH = os.path.join(OUT, "auxetic_metrics.csv")

# columns identical to stage3_metrics_8040.csv
FIELDS = ["filename", "volume_fraction", "symmetry_class", "symmetry_score",
          "mirror_x", "mirror_y", "diagonal", "n_solid_components",
          "largest_component_fraction", "percolates_x", "percolates_y",
          "periodic_connected_x", "periodic_connected_y", "min_feature_px",
          "thin_solid_fraction", "thin_void_fraction",
          "floating_solid_fraction", "trapped_void_fraction",
          "est_min_solid_thickness_px", "manufacturable",
          "E_eff_x_rel", "E_eff_y_rel", "poisson_eff", "G_eff_rel",
          "K_eff_rel", "auxetic", "specific_stiffness_rel"]


def draw_rot_squares(m, alphas, s_frac, hinge_r):
    """Binary m x m cell: 2x2 alternately rotated squares + hinge disks."""
    h = m / 2.0
    s = h * s_frac / 2.0
    yy, xx = np.mgrid[0:m, 0:m]
    px, py = xx + 0.5, yy + 0.5
    solid = np.zeros((m, m), dtype=bool)
    centers = [(h/2, h/2), (3*h/2, h/2), (h/2, 3*h/2), (3*h/2, 3*h/2)]
    for k, (cx, cy) in enumerate(centers):
        rot = alphas[k]
        for ox in (-m, 0, m):
            for oy in (-m, 0, m):
                dx, dy = px - (cx + ox), py - (cy + oy)
                u = np.cos(rot) * dx + np.sin(rot) * dy
                v = -np.sin(rot) * dx + np.cos(rot) * dy
                solid |= np.maximum(np.abs(u), np.abs(v)) <= s
    pts = [(h, h/2), (h/2, h), (0, h/2), (h/2, 0),
           (h, 3*h/2), (3*h/2, h), (3*h/2, 0), (0, 3*h/2)]
    for hx, hy in pts:
        for ox in (-m, 0, m):
            for oy in (-m, 0, m):
                solid |= np.hypot(px - (hx + ox), py - (hy + oy)) <= hinge_r
    return solid.astype(np.uint8)


def cell_for_vf(m, alphas, hinge_r, vf_target):
    """Bisect the square size so the cell hits the target volume fraction."""
    lo, hi = 0.3, 1.4
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if draw_rot_squares(m, alphas, mid, hinge_r).mean() < vf_target:
            lo = mid
        else:
            hi = mid
    return draw_rot_squares(m, alphas, 0.5 * (lo + hi), hinge_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(IMG_DIR, exist_ok=True)

    rows, seen = [], set()
    made = attempts = 0
    while made < args.n and attempts < args.n * 30:
        attempts += 1
        # alternating-sign rotations with small per-square jitter
        base = rng.uniform(0.20, 0.62)
        signs = np.array([1, -1, -1, 1])
        alphas = signs * (base + rng.normal(0, 0.03, 4))
        hinge_r = rng.uniform(1.5, 2.5)
        vf_t = rng.uniform(0.34, 0.62)
        a = cell_for_vf(M, alphas, hinge_r, vf_t)
        # NOTE: cells are kept ALIGNED (no periodic roll). The hinge midpoints
        # then sit at fixed lattice points, which the downstream hinge-finishing
        # post-process and the nu conditioning both rely on.
        key = hashlib.md5(a.tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        big = np.kron(a, np.ones((2, 2), dtype=np.uint8))     # 32 -> 64
        met = compute_metrics(big, do_fem=True)
        if "fem_error" in met or not (met["periodic_connected_x"]
                                      and met["periodic_connected_y"]):
            continue
        fname = f"aux_{made:04d}.png"
        Image.fromarray((big * 255).astype(np.uint8), mode="L").save(
            os.path.join(IMG_DIR, fname))
        met.pop("CH", None)
        rows.append({"filename": fname, **{k: met[k] for k in FIELDS[1:]}})
        made += 1
        if made % 100 == 0:
            nu = np.array([float(r["poisson_eff"]) for r in rows])
            print(f"{made}/{args.n}  nu: min {nu.min():.3f} "
                  f"median {np.median(nu):.3f}  auxetic {(nu < -0.01).mean():.0%}",
                  flush=True)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    nu = np.array([float(r["poisson_eff"]) for r in rows])
    print(f"\nwrote {len(rows)} cells to {IMG_DIR}")
    print(f"labels at {CSV_PATH}")
    print(f"nu: min {nu.min():.3f}  p10 {np.percentile(nu,10):.3f}  "
          f"median {np.median(nu):.3f}  p90 {np.percentile(nu,90):.3f}")
    print(f"auxetic (<-0.01): {(nu < -0.01).sum()} "
          f"({(nu < -0.01).mean():.0%}), strong (<-0.3): {(nu < -0.3).sum()}")


if __name__ == "__main__":
    main()
