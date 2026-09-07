"""
Fair 'full topology optimization from scratch' baseline for max bulk modulus.

The earlier femlib.full_to checkerboarded (fine composite optimum that fragments
on 0/1 projection). This is a proper robust-projection SIMP:
  - density filter (min length scale)                         [Bruns & Tortorelli]
  - smooth Heaviside PROJECTION with beta-continuation        [Guest 2004;
    Wang, Lazarov & Sigmund 2011] -> drives to clean 0/1, kills checkerboards
  - OC update, volume constraint on the projected field
  - inverse-homogenization max-bulk objective                 [Sigmund 1994]
  - runs at higher resolution (48/64) so features resolve.

Gives a legitimate cold-start reference to compare against the diffusion
warm start. Uses femlib's KE/CHI0/solver so it's the same physics.

Usage:  python full_to_baseline.py            # sweeps 32/48/64 at vf 0.5
"""
import numpy as np
from scipy import sparse
import femlib as fl


def _proj(xt, beta, eta=0.5):
    """Smooth Heaviside projection and its derivative wrt xt."""
    tb = np.tanh(beta * eta)
    num = tb + np.tanh(beta * (xt - eta))
    den = tb + np.tanh(beta * (1 - eta))
    xphys = num / den
    dxp = beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / den
    return xphys, dxp


def cold_to(m, vf, rmin=None, iters=150, p=3, move=0.2, Emin=1e-3, seed=0,
            eta=0.5, beta0=1.0, beta_max=32.0, beta_double_every=25):
    """Robust-projection cold-start max-bulk TO. Returns (binary, info)."""
    if rmin is None:
        rmin = max(2.0, m / 16.0)                     # physical length ~ constant
    nel = m * m
    edof = fl._edof(m)
    H = fl._filter_matrix(m, rmin); HT = H.T.tocsr()
    rng = np.random.default_rng(seed)
    # small structured + random seed to break symmetry (projection cleans it up)
    x = np.clip(vf + 0.05 * (rng.random(nel) - 0.5), 0.01, 0.99)
    beta = beta0
    bulk_hist = []; traj = []
    for it in range(iters):
        if it > 0 and it % beta_double_every == 0:
            beta = min(beta_max, beta * 2)
        xt = H.dot(x)
        xphys, dxp = _proj(xt, beta, eta)
        scale = Emin + (1 - Emin) * xphys ** p
        dscale = (1 - Emin) * p * xphys ** (p - 1)
        chi = fl._solve_chi(scale.astype(float), m, edof)
        E00, E11, E01 = fl._element_energies(chi, m, edof)
        V = nel
        bulk = (np.sum(scale * E00) + np.sum(scale * E11) + 2 * np.sum(scale * E01)) / 4.0 / V
        bulk_hist.append(bulk)
        # minimize c = -bulk ; dc/dxphys, then chain through projection + filter
        dc_dxphys = -dscale * (E00 + E11 + 2 * E01) / 4.0 / V
        dc = HT.dot(dc_dxphys * dxp)
        dvol_dxphys = np.full(nel, 1.0 / nel)
        dv = HT.dot(dvol_dxphys * dxp)
        # OC with volume constraint on projected field
        l1, l2 = 1e-9, 1e9; xn = x
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lm = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dc / (dv * lm + 1e-30))
            xn = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(be)))))
            xphys_n, _ = _proj(H.dot(xn), beta, eta)
            if xphys_n.mean() > vf: l1 = lm
            else: l2 = lm
        x = xn
        if (it + 1) % 10 == 0 or it == iters - 1:
            xb = (_proj(H.dot(x), beta, eta)[0] >= 0.5).astype(np.uint8)
            traj.append((3 * (it + 1), fl.homogenize(xb, m)["E"], fl.connectivity(xb, m)["connected"]))
    xphys, _ = _proj(H.dot(x), beta, eta)
    binary = (xphys >= 0.5).astype(np.uint8)
    h = fl.homogenize(binary, m); c = fl.connectivity(binary, m)
    return binary, {"E": h["E"], "bulk": h["bulk"], "nu": h["nu"], "vf": fl.volume_fraction(binary),
                    "pieces": c["pieces"], "connected": c["connected"],
                    "fe_solves": 3 * iters, "bulk_hist": bulk_hist, "traj": traj, "rmin": rmin, "res": m}


if __name__ == "__main__":
    print("Fair cold-start full TO (robust Heaviside projection), max bulk, vf 0.50")
    print("res  rmin  iters  E      bulk    nu      pieces  conn   FEsolves")
    for m in (32, 48, 64):
        b, info = cold_to(m, 0.5, iters=150, seed=0)
        print(f"{m:3d}  {info['rmin']:.1f}   150   {info['E']:.3f}  {info['bulk']:.3f}  "
              f"{info['nu']:+.3f}  {info['pieces']:4d}   {str(info['connected']):5s}  {info['fe_solves']}")
