"""Auxetic SIMP: beta + iteration-budget sweep.

Finding from harness_aux: the deployed auxetic polish (beta=0.5) drives nu POSITIVE
(destroys auxeticity) and is discarded by the keep-if-better rule -> 30 iters wasted.
Root cause = the stiffness-penalty weight beta. This sweeps beta and the iteration
budget to find (a) a beta that actually deepens nu while keeping cells connected,
and (b) the iteration count where nu-depth plateaus. Tracks nu, E, connectivity,
pieces. One sample batch (frozen specialist), so beta is compared on the same cells.
"""
import os, csv, numpy as np
import femlib as fl
import ddpm_aux as da

HERE = os.path.dirname(os.path.abspath(__file__)); M = 32; MIN = 6
m = da.load_model(os.path.join(HERE, "checkpoints", "ddpmaux_ep120.pt"))
N = 16
cells = da.ddim_sample(m, N, da.make_cond(0.47, -0.5), guidance=3.0, seed=2000)
base = []
for c in cells:
    cc = fl.despeckle(c, M, MIN)
    h = fl.homogenize(cc, M); con = fl.connectivity(cc, M)
    base.append((cc, h["nu"], h["E"], con["connected"], fl.volume_fraction(cc)))
nu0 = np.median([b[1] for b in base])
print(f"raw specialist (despeckled), n={N}: nu(med) {nu0:+.3f}  %aux {100*np.mean([b[1]<-0.01 for b in base]):.0f}%  "
      f"conn {100*np.mean([b[3] for b in base]):.0f}%  E(med) {np.median([b[2] for b in base]):.3f}")

ITERS = 30
rows = []
for beta in (0.0, 0.1, 0.2, 0.5):
    trajs = []
    for cc, nu, E, con, vf in base:
        trajs.append(fl.optimize_cell(cc, M, vf, iters=ITERS, rmin=1.5,
                                      objective="auxetic", beta=beta, symmetric=True))
    for k in range(1, ITERS + 1):
        nus = []; Es = []; conns = []; pcs = []; kept = []
        for (cc, nu, E, con, vf), tr in zip(base, trajs):
            pc = fl.despeckle(tr[k - 1], M, MIN)
            h = fl.homogenize(pc, M); ci = fl.connectivity(pc, M)
            nus.append(h["nu"]); Es.append(h["E"]); conns.append(ci["connected"]); pcs.append(ci["pieces"])
            kept.append(min(nu, h["nu"]))            # demo keeps polish only if nu improves
        rows.append({"beta": beta, "k": k, "fe": 3 * k,
                     "nu_med": float(np.median(nus)), "nu_kept_med": float(np.median(kept)),
                     "E_med": float(np.median(Es)), "conn": float(np.mean(conns)),
                     "pieces": float(np.mean(pcs)), "aux": float(np.mean(np.array(nus) < -0.01))})

with open(os.path.join(HERE, "study_aux_beta.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def row(beta, k): return next(r for r in rows if r["beta"] == beta and r["k"] == k)
print("\nbeta  k    nu(med)  nu(kept)  E(med)  conn  pieces  %aux")
for beta in (0.0, 0.1, 0.2, 0.5):
    for k in (1, 2, 3, 5, 10, 20, 30):
        r = row(beta, k)
        print(f"{beta:.1f}  {k:2d}   {r['nu_med']:+.3f}   {r['nu_kept_med']:+.3f}   {r['E_med']:.3f}  "
              f"{100*r['conn']:3.0f}%  {r['pieces']:4.1f}  {100*r['aux']:3.0f}%")
    print()
# plateau (95% of converged depth) for the best beta candidates
for beta in (0.0, 0.1, 0.2):
    conv = row(beta, ITERS)["nu_med"]
    kstar = next((k for k in range(1, ITERS + 1) if row(beta, k)["nu_med"] <= 0.95 * conv), None)
    print(f"beta {beta}: converged nu(med) {conv:+.3f}, 95% depth by k={kstar} ({3*(kstar or 0)} FE), "
          f"conn {100*row(beta, kstar or ITERS)['conn']:.0f}%")
