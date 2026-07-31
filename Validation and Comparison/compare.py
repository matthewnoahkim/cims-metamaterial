"""
Validation and Comparison.

Compares four design sources at target volume fractions, scoring every cell with
the SAME Stage 3 evaluator (8-connectivity, full periodic homogenization at 32x32):
  - random   : Gaussian-smoothed noise thresholded to the exact target vf
  - manual   : parametric cells from the dataset near the target vf
  - vae      : the Stage 4 conditional VAE, conditioned on the target
  - diffusion: the Stage 5 v2 DDPM (calibrated request + rejection by the evaluator)

Writes comparison_all_cells.csv (one row per cell), prints the summary table, and
saves a comparison figure per target.

Usage: python compare.py            # targets 0.40 and 0.50, 16 cells each
"""
import os, csv, glob
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
from metrics import compute_metrics
from cvae import load as load_vae, sample as vae_sample, make_cond as vae_cond
from ddpm2 import UNet, ddim_sample, make_cond as dif_cond, DEVICE

TARGETS = [0.40, 0.50]
N = 16
OUT_CSV = "comparison_all_cells.csv"
CAL_REQ = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
CAL_ACH = np.array([0.367, 0.400, 0.442, 0.488, 0.529, 0.567])
req_for = lambda t: float(np.interp(np.clip(t, CAL_ACH[0], CAL_ACH[-1]), CAL_ACH, CAL_REQ))
rng = np.random.default_rng(0)


def to32(img64):
    im = Image.fromarray((img64 * 255).astype(np.uint8)).resize((32, 32), Image.BILINEAR)
    return (np.array(im) > 127).astype(np.uint8)


def random_pattern(vf, n=32):
    z = gaussian_filter(rng.standard_normal((n, n)), 1.5)
    return (z >= np.quantile(z, 1 - vf)).astype(np.uint8)


def build_groups(target, vae_model, dif_model, manual_pool):
    g = {}
    g["random"] = [random_pattern(target) for _ in range(N)]
    near = [a for a in manual_pool if abs(a.mean() - target) <= 0.04]
    g["manual"] = [near[i] for i in rng.choice(len(near), N, replace=False)]
    # VAE: condition on target vf, high stiffness, biaxial symmetry (64 -> 32)
    v64 = vae_sample(vae_model, N, vae_cond(target, "high", "biaxial"), symmetrize=True)
    g["vae"] = [to32(c) for c in v64]
    # diffusion: calibrated request + rejection for connectivity & vf tolerance
    kept, tries = [], 0
    while len(kept) < N and tries < 10:
        for c in ddim_sample(dif_model, N, dif_cond(req_for(target), "high"),
                             steps=40, guidance=3.0, seed=100 + tries):
            mm = compute_metrics(c, do_fem=False)
            if mm["periodic_connected_x"] and mm["periodic_connected_y"] \
                    and abs(c.mean() - target) <= 0.05:
                kept.append(c)
        tries += 1
    g["diffusion"] = kept[:N]
    g["_dif_yield"] = (len(kept), N * tries)
    return g


def score_rows(cells, group, target):
    out = []
    for a in cells:
        m = compute_metrics(np.asarray(a).reshape(32, 32))
        out.append({
            "group": group, "target_vf": target,
            "volume_fraction": m["volume_fraction"],
            "n_solid_components": m["n_solid_components"],
            "connected": int(m["periodic_connected_x"] and m["periodic_connected_y"]),
            "symmetry_score": m["symmetry_score"],
            "E_eff_x_rel": m["E_eff_x_rel"], "E_eff_y_rel": m["E_eff_y_rel"],
            "poisson_eff": m["poisson_eff"],
        })
    return out


def summarize(rows, target):
    print(f"\n=== target vf {target} ===")
    print(f"{'group':12s} {'n':>3s} {'vf_err':>7s} {'comps':>6s} {'conn':>5s} {'sym':>6s} {'E_eff':>7s}")
    for grp in ["random", "manual", "vae", "diffusion"]:
        r = [x for x in rows if x["group"] == grp and x["target_vf"] == target]
        if not r:
            continue
        vfe = np.mean([abs(x["volume_fraction"] - target) for x in r])
        comps = np.mean([x["n_solid_components"] for x in r])
        conn = np.mean([x["connected"] for x in r])
        sym = np.mean([x["symmetry_score"] for x in r])
        E = np.mean([0.5 * (x["E_eff_x_rel"] + x["E_eff_y_rel"]) for x in r])
        print(f"{grp:12s} {len(r):3d} {vfe:7.3f} {comps:6.1f} {conn:5.2f} {sym:6.3f} {max(0,E):7.4f}")


def figure(groups, target, path):
    order = ["random", "manual", "vae", "diffusion"]
    S, cols, pad = 32, 8, 2
    W = cols * S + (cols + 1) * pad
    H = len(order) * S + (len(order) + 1) * pad
    cv = np.full((H, W), 170, np.uint8)
    for r, grp in enumerate(order):
        for i, a in enumerate(groups[grp][:cols]):
            a = np.asarray(a).reshape(S, S)
            y = pad + r * (S + pad); x = pad + i * (S + pad)
            cv[y:y + S, x:x + S] = a * 255
    Image.fromarray(cv).resize((W * 4, H * 4), Image.NEAREST).save(path)


def main():
    vae_model, _ = load_vae("cvae")
    dck = torch.load("checkpoints/ddpm2_ep25.pt", map_location=DEVICE, weights_only=False)
    dif_model = UNet().to(DEVICE); dif_model.load_state_dict(dck["ema"])
    manual_pool = [(np.array(Image.open(f).convert("L").resize((32, 32), Image.BILINEAR)) > 127).astype(np.uint8)
                   for f in glob.glob("unitcells_big/images/*.png")[:3000]]

    all_rows = []
    for t in TARGETS:
        g = build_groups(t, vae_model, dif_model, manual_pool)
        for grp in ["random", "manual", "vae", "diffusion"]:
            all_rows += score_rows(g[grp], grp, t)
        summarize(all_rows, t)
        print(f"diffusion rejection yield: {g['_dif_yield'][0]}/{g['_dif_yield'][1]}")
        figure(g, t, f"fig_compare_vf{t}.png")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
