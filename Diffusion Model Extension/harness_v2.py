"""
Attribution / correction-budget study v2 — improvements found from v1.

Fixes v1's weaknesses and implements the pipeline improvements its findings imply:
  1. Fair cold baseline  — uniform+noise init, 40 iters, best of R restarts (real
     "full TO from scratch", converged), replacing v1's under-converged 15-iter run.
  2. Continuous warm start — SIMP initialized from the diffusion model's CONTINUOUS
     density x0 (not the re-binarized 0.9/0.1 proxy). Better init, cleaner test.
  3. Guidance -> SIMP synergy — a fourth arm: SIMP warm-started from a physics-GUIDED
     continuous sample, to test whether guidance (the big model-side win in v1) cuts
     the SIMP budget needed.
  4. Uncertainty — IQR bands, larger n, and an explicit "budget to reach 95%" readout.

Model frozen (ddpm2 EMA ep25). One evaluator (femlib). Outputs *_v2 files here.

Usage:  python harness_v2.py --n 20 --targets 0.40 0.50 --warm-iters 15 --cold-iters 40 --restarts 2
"""
import os, csv, json, argparse
import numpy as np
import torch
import femlib as fl
import ddpm2

DEVICE = ddpm2.DEVICE
IMG, T = 32, ddpm2.T_STEPS
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints", "ddpm2_ep25.pt")
CAL_REQ = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
CAL_ACH = np.array([0.367, 0.400, 0.442, 0.488, 0.529, 0.567])
def request_for(t): return float(np.interp(np.clip(t, CAL_ACH[0], CAL_ACH[-1]), CAL_ACH, CAL_REQ))


def load_model():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    m = ddpm2.UNet().to(DEVICE); m.load_state_dict(ck["ema"]); m.eval()
    return m


@torch.no_grad()
def sample(model, n, cond_vec, steps=40, guidance=3.0, seed=0, guide=None):
    """DDIM+CFG (optionally physics-guided). Returns (binaries, densities) where
    density = (x0_final+1)/2 in [0,1] — the model's continuous field before threshold."""
    torch.manual_seed(seed)
    ab = ddpm2.abar.to(DEVICE)
    ts = [int((T - 1) * (1 - i / (steps - 1))) for i in range(steps)]
    x = torch.randn(n, 1, IMG, IMG, device=DEVICE)
    c = torch.tensor(cond_vec, dtype=torch.float32, device=DEVICE).expand(n, len(cond_vec))
    x0 = x
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
    xn = x.cpu().numpy()
    bins = [((xn[i, 0].ravel() + 1) / 2 >= 0.5).astype(np.uint8) for i in range(n)]
    dens = [np.clip((xn[i, 0].ravel() + 1) / 2, 0, 1) for i in range(n)]
    return bins, dens


def ev(cell, target):
    h = fl.homogenize(cell, IMG); c = fl.connectivity(cell, IMG)
    vf = fl.volume_fraction(cell)
    return {"E": h["E"], "bulk": h["bulk"], "nu": h["nu"], "vf": vf,
            "vferr": abs(vf - target), "pieces": c["pieces"], "connected": int(c["connected"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.40, 0.50])
    ap.add_argument("--warm-iters", type=int, default=15)
    ap.add_argument("--cold-iters", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--min-size", type=int, default=6)
    args = ap.parse_args()
    model = load_model(); MIN = args.min_size
    rows = []
    def add(target, seed, arm, k, cell):
        m = ev(cell, target)
        rows.append({"target": target, "seed": seed, "arm": arm, "k": k, "fe": 3 * k, **m})

    guideK = min(12, max(4, round(args.steps * 0.3)))
    for target in args.targets:
        req = request_for(target); cond = ddpm2.make_cond(req, "high")
        print(f"\n=== target vf {target} (req {req:.3f}) ===", flush=True)
        Db, Dd = sample(model, args.n, cond, steps=args.steps, seed=1000)
        Gb, Gd = sample(model, args.n, cond, steps=args.steps, seed=1000,
                        guide={"K": guideK, "strength": 0.25, "objective": "bulk", "beta": 0.5})
        for i in range(args.n):
            vfd = fl.volume_fraction(Db[i])
            add(target, i, "diffusion_raw", 0, Db[i])
            add(target, i, "guided_raw", 0, Gb[i])
            # warm (binarized 0.9/0.1) — v1 method, for comparison
            for k, c in enumerate(fl.optimize_cell(fl.despeckle(Db[i], IMG, MIN), IMG, vfd,
                                   iters=args.warm_iters, rmin=2.0), start=1):
                add(target, i, "warm_bin", k, fl.despeckle(c, IMG, MIN))
            # warm (continuous diffusion density) — improved init
            for k, c in enumerate(fl.optimize_cell(Db[i], IMG, vfd, iters=args.warm_iters,
                                   rmin=2.0, init_density=Dd[i]), start=1):
                add(target, i, "warm_cont", k, fl.despeckle(c, IMG, MIN))
            # warm from a GUIDED continuous sample — guidance -> SIMP synergy
            vfg = fl.volume_fraction(Gb[i])
            for k, c in enumerate(fl.optimize_cell(Gb[i], IMG, vfg, iters=args.warm_iters,
                                   rmin=2.0, init_density=Gd[i]), start=1):
                add(target, i, "warm_guided", k, fl.despeckle(c, IMG, MIN))
            # fair cold TO: uniform+noise, best of R restarts, long budget
            best = None
            rng = np.random.default_rng(100 + i)
            for r in range(args.restarts):
                init = np.clip(vfd + 0.05 * (rng.random(IMG * IMG) - 0.5), 0.02, 0.98)
                traj = fl.optimize_cell(Db[i], IMG, vfd, iters=args.cold_iters, rmin=2.0,
                                        init_density=init)
                fE = ev(fl.despeckle(traj[-1], IMG, MIN), target)["E"]
                if best is None or fE > best[0]:
                    best = (fE, traj)
            for k, c in enumerate(best[1], start=1):
                add(target, i, "cold", k, fl.despeckle(c, IMG, MIN))
        print(f"  done {args.n} seeds", flush=True)

    with open(os.path.join(HERE, "study_results_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target","seed","arm","k","fe","E","bulk","nu","vf","vferr","pieces","connected"])
        w.writeheader(); w.writerows(rows)
    print("\nwrote study_results_v2.csv (%d rows)" % len(rows))

    # quick console readout
    def col(arm, k, key="E"):
        return np.array([r[key] for r in rows if r["arm"] == arm and r["k"] == k])
    print("\narm                 E(med)  conn   pieces")
    for a in ["diffusion_raw", "guided_raw"]:
        e = col(a, 0); print(f"  {a:16s} k0  {np.median(e):.3f}  {100*col(a,0,'connected').mean():4.0f}%  {col(a,0,'pieces').mean():4.1f}")
    for a in ["warm_bin", "warm_cont", "warm_guided", "cold"]:
        kk = args.cold_iters if a == "cold" else args.warm_iters
        e = col(a, kk); print(f"  {a:16s} k{kk:<2d} {np.median(e):.3f}  {100*col(a,kk,'connected').mean():4.0f}%  {col(a,kk,'pieces').mean():4.1f}")


if __name__ == "__main__":
    main()
