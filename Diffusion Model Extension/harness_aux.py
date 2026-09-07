"""
Auxetic-path budget study — does the auxetic SIMP polish need all 30 iters?

Mirrors harness_v2 for the auxetic path: specialist model (ddpm_aux, cond=[vf,nu]),
SIMP objective = minimize C1122 - beta*(C1111+C2222) (Sigmund 1995) at rmin=1.5,
symmetric, beta=0.5, and the metric is Poisson's ratio nu (more negative = better).
Init matches the DEPLOYED path (binary 0.9/0.1 from the despeckled sample), so the
"trim iters" recommendation applies to the current pipeline. One evaluator (femlib).

Arms: diffusion_raw, guided_raw (auxetic physics guidance), warm (SIMP from sample),
warm_guided (SIMP from a guided sample). Reports nu vs correction budget + the
iteration at which auxetic depth plateaus.

Usage:  python harness_aux.py --n 16 --targets -0.5 -0.3 --iters 30
"""
import os, csv, json, argparse
import numpy as np
import torch
import femlib as fl
import ddpm_aux as da

DEVICE = da.DEVICE
IMG, T = 32, da.T_STEPS
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints", "ddpmaux_ep120.pt")
MIN = 6


def load_model():
    return da.load_model(CKPT)


@torch.no_grad()
def sample(model, n, cond_vec, steps=40, guidance=3.0, seed=0, guide=None):
    """DDIM+CFG for the auxetic specialist, optionally auxetic-guided on the last
    guide['K'] steps (target-aware: stop pushing a cell once nu <= target)."""
    torch.manual_seed(seed)
    ab = da.abar.to(DEVICE)
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
                g, nu = fl.stiffness_grad(dens, IMG, objective="auxetic", beta=guide.get("beta", 0.5))
                if guide.get("target") is not None and nu <= guide["target"]:
                    continue
                gm = g.mean(); mx = np.max(np.abs(g - gm)) + 1e-12
                v = xn[i, 0].ravel() + guide["strength"] * (g - gm) / mx
                xn[i, 0] = np.clip(v, -1, 1).reshape(IMG, IMG)
            x0 = torch.from_numpy(xn).to(DEVICE)
        aP = ab[ts[s + 1]] if s < steps - 1 else torch.tensor(0.0, device=DEVICE)
        x = x0 if s == steps - 1 else aP.sqrt() * x0 + (1 - aP).sqrt() * eps
    xn = x.cpu().numpy()
    return [((xn[i, 0].ravel() + 1) / 2 >= 0.5).astype(np.uint8) for i in range(n)]


def ev(cell):
    h = fl.homogenize(cell, IMG); c = fl.connectivity(cell, IMG)
    return {"nu": h["nu"], "E": h["E"], "vf": fl.volume_fraction(cell),
            "pieces": c["pieces"], "connected": int(c["connected"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--targets", type=float, nargs="+", default=[-0.5, -0.3])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    model = load_model()
    rows = []
    def add(target, seed, arm, k, cell):
        m = ev(cell); rows.append({"target": target, "seed": seed, "arm": arm, "k": k, "fe": 3 * k, **m})

    guideK = min(12, max(4, round(args.steps * 0.3)))
    for target in args.targets:
        nuReq = min(-0.2, max(-1.0, target))
        cond = da.make_cond(0.47, nuReq)
        print(f"\n=== target nu {target} (cond nu {nuReq}) ===", flush=True)
        D = sample(model, args.n, cond, steps=args.steps, seed=2000)
        G = sample(model, args.n, cond, steps=args.steps, seed=2000,
                   guide={"K": guideK, "strength": 0.25, "beta": 0.5, "target": target})
        for i in range(args.n):
            add(target, i, "diffusion_raw", 0, D[i])
            add(target, i, "guided_raw", 0, G[i])
            # warm SIMP (auxetic), binary init from despeckled sample — matches demo
            dd = fl.despeckle(D[i], IMG, MIN); vfd = fl.volume_fraction(dd)
            traj = fl.optimize_cell(dd, IMG, vfd, iters=args.iters, rmin=1.5,
                                    objective="auxetic", beta=0.5, symmetric=True)
            for k, c in enumerate(traj, start=1):
                add(target, i, "warm", k, fl.despeckle(c, IMG, MIN))
            # guided init + auxetic SIMP
            gd = fl.despeckle(G[i], IMG, MIN); vfg = fl.volume_fraction(gd)
            trajg = fl.optimize_cell(gd, IMG, vfg, iters=args.iters, rmin=1.5,
                                     objective="auxetic", beta=0.5, symmetric=True)
            for k, c in enumerate(trajg, start=1):
                add(target, i, "warm_guided", k, fl.despeckle(c, IMG, MIN))
        print(f"  done {args.n} seeds", flush=True)

    with open(os.path.join(HERE, "study_aux.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target","seed","arm","k","fe","nu","E","vf","pieces","connected"])
        w.writeheader(); w.writerows(rows)
    print("\nwrote study_aux.csv (%d rows)" % len(rows))

    def colnu(arm, k): return np.array([r["nu"] for r in rows if r["arm"] == arm and r["k"] == k])
    def med(arm, k):
        a = colnu(arm, k); return float(np.median(a)) if len(a) else np.nan
    def frac_aux(arm, k):
        a = colnu(arm, k); return float(np.mean(a < -0.01)) if len(a) else np.nan
    def conn(arm, k):
        a = [r["connected"] for r in rows if r["arm"] == arm and r["k"] == k]; return float(np.mean(a)) if a else np.nan
    print("\narm            k     nu(med)  %aux  conn")
    for a, k in [("diffusion_raw",0),("guided_raw",0)]:
        print(f"  {a:13s} {k:2d}   {med(a,k):+.3f}  {100*frac_aux(a,k):3.0f}%  {100*conn(a,k):3.0f}%")
    print("  -- warm (SIMP from sample) --")
    for k in [1,2,3,5,7,10,15,20,25,args.iters]:
        print(f"  warm          {k:2d}   {med('warm',k):+.3f}  {100*frac_aux('warm',k):3.0f}%  {100*conn('warm',k):3.0f}%   | warm_guided {med('warm_guided',k):+.3f} {100*frac_aux('warm_guided',k):3.0f}%")
    # convergence: iteration reaching 95% of converged auxetic DEPTH (most-negative)
    conv = med("warm", args.iters)
    kstar = next((k for k in range(1, args.iters + 1) if med("warm", k) <= 0.95 * conv), None)
    convg = med("warm_guided", args.iters)
    kstarg = next((k for k in range(1, args.iters + 1) if med("warm_guided", k) <= 0.95 * convg), None)
    print(f"\nconverged nu: warm {conv:+.3f} (95% depth by k={kstar}, {3*(kstar or 0)} FE);  "
          f"warm_guided {convg:+.3f} (95% by k={kstarg}, {3*(kstarg or 0)} FE)")
    json.dump({"conv_warm": conv, "kstar_warm": kstar, "conv_guided": convg, "kstar_guided": kstarg,
               "n": args.n, "targets": args.targets, "iters": args.iters},
              open(os.path.join(HERE, "study_aux_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
