"""
Attribution / correction-budget study (frozen pipeline).

Answers two coupled questions on the SAME runs, all scored by the one evaluator
in femlib (demo-matched: KE/CHI0, softened void 1e-3, native 32x32):

  (1) Attribution — how much of the improvement comes from the generative model,
      the despeckle step, and the SIMP polish, separately.
  (2) The scientific question — do diffusion samples reach good designs under
      only LIGHT physics correction, versus full topology optimization from
      scratch? We compare, at matched correction budget (FE solves), SIMP started
      from the diffusion sample (warm start) vs. from a uniform field (cold /
      full-TO start), plus a physics-guided arm.

Model is FROZEN (ddpm2 EMA epoch 25). No new model complexity — this is
measurement only. Outputs: study_results.csv, study_summary.json,
study_curves.png in this folder.

Usage:  python harness.py --n 12 --targets 0.40 0.50 --iters 15
"""
import os, csv, json, argparse
import numpy as np
import torch
import femlib as fl
import ddpm2

DEVICE = ddpm2.DEVICE
IMG, T = 32, ddpm2.T_STEPS
ABAR = ddpm2.abar.numpy()
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints", "ddpm2_ep25.pt")

# vf request calibration (achieved -> request), from EXPORT_NOTES / generate2
CAL_REQ = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
CAL_ACH = np.array([0.367, 0.400, 0.442, 0.488, 0.529, 0.567])
def request_for(target):
    return float(np.interp(np.clip(target, CAL_ACH[0], CAL_ACH[-1]), CAL_ACH, CAL_REQ))


def load_model():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    m = ddpm2.UNet().to(DEVICE); m.load_state_dict(ck["ema"]); m.eval()
    return m


@torch.no_grad()
def sample(model, n, cond_vec, steps=40, guidance=3.0, seed=0,
           guide=None):
    """DDIM + CFG, optionally physics-guided on the last guide['K'] steps
    (port of the demo's sampleBatch). Returns list of binary 32x32 cells."""
    torch.manual_seed(seed)
    ab = ddpm2.abar.to(DEVICE)
    ts = [int((T - 1) * (1 - i / (steps - 1))) for i in range(steps)]
    x = torch.randn(n, 1, IMG, IMG, device=DEVICE)
    c = torch.tensor(cond_vec, dtype=torch.float32, device=DEVICE).expand(n, len(cond_vec))
    for s in range(steps):
        t = torch.full((n,), ts[s], device=DEVICE, dtype=torch.long)
        ec = model(x, t, c); eu = model.forward_uncond(x, t)
        eps = eu + guidance * (ec - eu)
        a = ab[ts[s]]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        if guide and s >= steps - guide["K"]:
            xn = x0.cpu().numpy()
            for i in range(n):
                dens = (xn[i, 0].ravel() + 1) * 0.5
                g, _ = fl.stiffness_grad(dens, IMG, objective=guide.get("objective", "bulk"),
                                         beta=guide.get("beta", 0.5))
                gm = g.mean(); mx = np.max(np.abs(g - gm)) + 1e-12
                v = xn[i, 0].ravel() + guide["strength"] * (g - gm) / mx
                xn[i, 0] = np.clip(v, -1, 1).reshape(IMG, IMG)
            x0 = torch.from_numpy(xn).to(DEVICE)
        aP = ab[ts[s + 1]] if s < steps - 1 else torch.tensor(0.0, device=DEVICE)
        x = x0 if s == steps - 1 else aP.sqrt() * x0 + (1 - aP).sqrt() * eps
    out = []
    xn = x.cpu().numpy()
    for i in range(n):
        out.append(((xn[i, 0].ravel() + 1) / 2 >= 0.5).astype(np.uint8))
    return out


def ev(cell, target):
    h = fl.homogenize(cell, IMG); c = fl.connectivity(cell, IMG)
    vf = fl.volume_fraction(cell)
    return {"vf": vf, "vferr": abs(vf - target), "pieces": c["pieces"],
            "connected": int(c["connected"]), "E": h["E"], "bulk": h["bulk"], "nu": h["nu"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.40, 0.50])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--min-size", type=int, default=6)
    args = ap.parse_args()
    model = load_model()
    MIN = args.min_size
    rows = []

    def add(target, seed, arm, k, cell):
        m = ev(cell, target)
        rows.append({"target": target, "seed": seed, "arm": arm, "k": k,
                     "fe_solves": 3 * k, **m})

    for target in args.targets:
        req = request_for(target)
        cond = ddpm2.make_cond(req, "high")
        print(f"\n=== target vf {target} (request {req:.3f}, stiffness high) ===", flush=True)
        # one diffusion batch (no guidance) and one guided batch, shared seed base
        D = sample(model, args.n, cond, steps=args.steps, guidance=3.0, seed=1000)
        G = sample(model, args.n, cond, steps=args.steps, guidance=3.0, seed=1000,
                   guide={"K": min(12, max(4, round(args.steps * 0.3))),
                          "strength": 0.25, "objective": "bulk", "beta": 0.5})
        for i in range(args.n):
            d = D[i]
            vfd = fl.volume_fraction(d)                       # SIMP preserves the cell's volume
            add(target, i, "diffusion_raw", 0, d)             # (a) model alone
            dd = fl.despeckle(d, IMG, MIN)
            add(target, i, "diffusion_despeckle", 0, dd)      # (b) + solidify
            add(target, i, "guided_raw", 0, G[i])             # (d) guidance during sampling
            # (c) SIMP from the despeckled diffusion sample (WARM start)
            warm = fl.optimize_cell(dd, IMG, vfd, iters=args.iters, rmin=2.0, objective="bulk")
            for k, cell in enumerate(warm, start=1):
                add(target, i, "warm_simp", k, fl.despeckle(cell, IMG, MIN))
            # control: SIMP from a UNIFORM field at the same volume (COLD / full-TO start)
            cold = fl.optimize_cell(dd, IMG, vfd, iters=args.iters, rmin=2.0, objective="bulk",
                                    init_density=np.full(IMG * IMG, vfd))
            for k, cell in enumerate(cold, start=1):
                add(target, i, "cold_simp", k, fl.despeckle(cell, IMG, MIN))
            print(f"  seed {i:2d}: raw E {ev(d,target)['E']:.3f} pieces {ev(d,target)['pieces']:2d} "
                  f"-> warm@{args.iters} E {ev(warm[-1],target)['E']:.3f} "
                  f"| cold@{args.iters} E {ev(cold[-1],target)['E']:.3f}", flush=True)

    # ---- write raw rows
    keys = ["target", "seed", "arm", "k", "fe_solves", "vf", "vferr", "pieces",
            "connected", "E", "bulk", "nu"]
    with open(os.path.join(HERE, "study_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # ---- aggregate
    R = rows
    def agg(arm, k):
        sel = [r for r in R if r["arm"] == arm and r["k"] == k]
        if not sel: return None
        return {"E": float(np.mean([r["E"] for r in sel])),
                "bulk": float(np.mean([r["bulk"] for r in sel])),
                "pieces": float(np.mean([r["pieces"] for r in sel])),
                "conn": float(np.mean([r["connected"] for r in sel])),
                "vferr": float(np.mean([r["vferr"] for r in sel])),
                "nu": float(np.median([r["nu"] for r in sel])), "n": len(sel)}

    summary = {"config": {"n": args.n, "targets": args.targets, "iters": args.iters,
                          "steps": args.steps, "min_size": MIN, "evaluator": "femlib void=1e-3 native32",
                          "model": "ddpm2 ema ep25 (frozen)"},
               "attribution_k0": {a: agg(a, 0) for a in ["diffusion_raw", "diffusion_despeckle", "guided_raw"]},
               "warm_simp": {k: agg("warm_simp", k) for k in range(1, args.iters + 1)},
               "cold_simp": {k: agg("cold_simp", k) for k in range(1, args.iters + 1)}}
    with open(os.path.join(HERE, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console report
    a = summary["attribution_k0"]
    print("\n================ ATTRIBUTION (budget k = 0) ================")
    print("arm                    E      pieces  conn   vferr")
    for name in ["diffusion_raw", "diffusion_despeckle", "guided_raw"]:
        s = a[name]
        print(f"  {name:20s} {s['E']:.3f}  {s['pieces']:5.1f}  {s['conn']*100:4.0f}%  {s['vferr']:.3f}")
    print("\n================ CORRECTION-BUDGET CURVES ================")
    print(" k  FEsolves | warm: E    conn  pieces | cold: E    conn  pieces")
    for k in range(1, args.iters + 1):
        w, c = summary["warm_simp"][k], summary["cold_simp"][k]
        print(f"{k:2d}  {3*k:6d}   |  {w['E']:.3f}  {w['conn']*100:3.0f}%  {w['pieces']:4.1f}  "
              f"|  {c['E']:.3f}  {c['conn']*100:3.0f}%  {c['pieces']:4.1f}")

    # ---- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(range(0, args.iters + 1))
        fe = [3 * k for k in ks]
        d0 = a["diffusion_despeckle"]  # warm start begins from the despeckled sample (k=0)
        warmE = [d0["E"]] + [summary["warm_simp"][k]["E"] for k in range(1, args.iters + 1)]
        coldE = [ev_uniform(args.targets)] + [summary["cold_simp"][k]["E"] for k in range(1, args.iters + 1)]
        warmC = [d0["conn"]] + [summary["warm_simp"][k]["conn"] for k in range(1, args.iters + 1)]
        coldC = [0.0] + [summary["cold_simp"][k]["conn"] for k in range(1, args.iters + 1)]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(fe, warmE, "o-", color="#058DC7", label="warm start (diffusion)")
        ax[0].plot(fe, coldE, "s--", color="#B03A2E", label="cold start (uniform / full TO)")
        ax[0].axhline(a["diffusion_raw"]["E"], color="#999", ls=":", lw=1, label="diffusion raw")
        ax[0].set_xlabel("correction budget (FE solves)"); ax[0].set_ylabel("effective stiffness E")
        ax[0].set_title("Stiffness vs. correction budget"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(fe, [100*v for v in warmC], "o-", color="#058DC7", label="warm start (diffusion)")
        ax[1].plot(fe, [100*v for v in coldC], "s--", color="#B03A2E", label="cold start (uniform)")
        ax[1].set_xlabel("correction budget (FE solves)"); ax[1].set_ylabel("connectivity (% of samples)")
        ax[1].set_title("Connectivity vs. correction budget"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "study_curves.png"), dpi=150)
        print("\nwrote study_curves.png")
    except Exception as e:
        print("plot skipped:", e)


def ev_uniform(targets):
    # E of a bare uniform field at k=0 is ~0 (no structure); report 0 for the cold curve start
    return 0.0


if __name__ == "__main__":
    main()
