"""
Offline FEM + optimization library for the attribution / correction-budget study.

Faithful Python port of the deployed demo's JavaScript (Metamaterial Generator/
index.html): the same periodic-homogenization element stiffness KE / unit-strain
fields CHI0, the same softened void (voidScale = 1e-3) evaluated natively at
32x32, the same SIMP inverse-homogenization polish (bulk-modulus objective, OC
update, cone density filter, Heaviside projection), the same guidance gradient,
and the same despeckle. One evaluator scores every arm so the ablation is fair.

Everything here is deterministic given its inputs; the generative model lives in
harness.py. Validated against analytic cases and the demo in __main__.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized

# ---- element stiffness (plane stress, nu0 = 0.3, unit square) and unit-strain fields
KE = np.array([
 [0.494505,0.178571,-0.302198,-0.013736,-0.247253,-0.178571,0.054945,0.013736],
 [0.178571,0.494505,0.013736,0.054945,-0.178571,-0.247253,-0.013736,-0.302198],
 [-0.302198,0.013736,0.494505,-0.178571,0.054945,-0.013736,-0.247253,0.178571],
 [-0.013736,0.054945,-0.178571,0.494505,0.013736,-0.302198,0.178571,-0.247253],
 [-0.247253,-0.178571,0.054945,0.013736,0.494505,0.178571,-0.302198,-0.013736],
 [-0.178571,-0.247253,-0.013736,-0.302198,0.178571,0.494505,0.013736,0.054945],
 [0.054945,-0.013736,-0.247253,0.178571,-0.302198,0.013736,0.494505,-0.178571],
 [0.013736,-0.302198,0.178571,-0.247253,-0.013736,0.054945,-0.178571,0.494505]])
CHI0 = np.array([
 [-0.5,0,-0.5],[0,-0.5,0],[0.5,0,-0.5],[0,-0.5,0],
 [0.5,0,0.5],[0,0.5,0],[-0.5,0,0.5],[0,0.5,0]], dtype=float)   # (8,3)
FE = KE @ CHI0                                                   # (8,3)

_EDOF_CACHE = {}

def _edof(m):
    """Periodic element->dof map, node order BL,BR,TR,TL (matches the JS)."""
    if m in _EDOF_CACHE:
        return _EDOF_CACHE[m]
    nel = m * m
    edof = np.empty((nel, 8), dtype=np.int64)
    for ey in range(m):
        for ex in range(m):
            e = ey * m + ex
            def nid(ix, iy): return (iy % m) * m + (ix % m)
            nodes = [nid(ex, ey), nid(ex + 1, ey), nid(ex + 1, ey + 1), nid(ex, ey + 1)]
            for k in range(4):
                edof[e, 2 * k] = 2 * nodes[k]
                edof[e, 2 * k + 1] = 2 * nodes[k] + 1
    _EDOF_CACHE[m] = edof
    return edof

def _assemble(scale, m, edof):
    """Global periodic K for element scales `scale` (nel,)."""
    ndof = 2 * m * m
    sK = (KE.ravel()[None, :] * scale[:, None]).ravel()
    iK = np.repeat(edof, 8, axis=1).ravel()
    jK = np.tile(edof, (1, 8)).ravel()
    K = sparse.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    return K

def _solve_chi(scale, m, edof, solve=None):
    """Solve K chi_c = F_c for the 3 unit-strain load cases; fix dofs 0,1."""
    ndof = 2 * m * m
    F = np.zeros((ndof, 3))
    contrib = scale[:, None, None] * FE[None, :, :]        # (nel,8,3)
    for c in range(3):
        np.add.at(F[:, c], edof.ravel(), contrib[:, :, c].ravel())
    free = np.arange(2, ndof)
    chi = np.zeros((ndof, 3))
    if solve is None:
        K = _assemble(scale, m, edof)
        chi[free] = spsolve(K[free][:, free], F[free]).reshape(len(free), 3)
    else:
        chi[free] = solve(F[free]).reshape(len(free), 3)
    return chi

def _element_energies(chi, m, edof):
    """Per-element mutual energies E00,E11,E01 of the 3 unit-strain solutions."""
    chi_e = chi[edof]                                      # (nel,8,3)
    d = CHI0.T[None, :, :] - np.transpose(chi_e, (0, 2, 1))  # (nel,3,8)
    dK = np.einsum('nck,kl->ncl', d, KE)                   # (nel,3,8)
    E = np.einsum('ncl,ndl->ncd', dK, d)                   # (nel,3,3)
    return E[:, 0, 0], E[:, 1, 1], E[:, 0, 1]

def homogenize(solid, m=None, void=1e-3):
    """Effective properties of a binary cell (solid=1/void=0), demo settings."""
    a = np.asarray(solid).ravel()
    if m is None:
        m = int(round(np.sqrt(a.size)))
    edof = _edof(m)
    scale = np.where(a > 0, 1.0, void).astype(float)
    chi = _solve_chi(scale, m, edof)
    E00, E11, E01 = _element_energies(chi, m, edof)
    V = m * m
    C00 = float(np.sum(scale * E00) / V)
    C11 = float(np.sum(scale * E11) / V)
    C01 = float(np.sum(scale * E01) / V)
    det = C00 * C11 - C01 * C01
    Ex = Ey = nu = 0.0
    if abs(det) > 1e-12:
        S00, S11, S01 = C11 / det, C00 / det, -C01 / det
        Ex, Ey, nu = 1.0 / S00, 1.0 / S11, -S01 / S00
    bulk = (C00 + C11 + 2 * C01) / 4.0
    return {"Ex": Ex, "Ey": Ey, "E": 0.5 * (Ex + Ey), "nu": nu, "bulk": bulk,
            "C00": C00, "C11": C11, "C01": C01}

# ------------------------------- connectivity / vf / despeckle (port of JS)
def volume_fraction(a):
    return float(np.mean(a))

def _uf_components(a, m):
    """Periodic 4-connected union-find; returns labels, sizes, percolation."""
    par = np.arange(m * m)
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def uni(x, y): par[find(x)] = find(y)
    A = a.ravel()
    for y in range(m):
        for x in range(m):
            if not A[y * m + x]:
                continue
            rx = (x + 1) % m; dy = (y + 1) % m
            if A[y * m + rx]: uni(y * m + x, y * m + rx)
            if A[dy * m + x]: uni(y * m + x, dy * m + x)
    roots = {}
    for i in range(m * m):
        if A[i]:
            r = find(i); roots[r] = roots.get(r, 0) + 1
    percX = percY = False
    for y in range(m):
        if A[y * m + 0] and A[y * m + (m - 1)] and find(y * m + 0) == find(y * m + (m - 1)):
            percX = True
    for x in range(m):
        if A[0 * m + x] and A[(m - 1) * m + x] and find(0 * m + x) == find((m - 1) * m + x):
            percY = True
    return par, roots, percX, percY

def connectivity(a, m=None):
    a = np.asarray(a).ravel().astype(np.uint8)
    if m is None:
        m = int(round(np.sqrt(a.size)))
    _, roots, percX, percY = _uf_components(a, m)
    return {"pieces": len(roots), "percX": percX, "percY": percY,
            "connected": bool(percX and percY)}

def despeckle(a, m=None, min_size=6):
    """Drop solid components smaller than min_size; fill lone void pinholes."""
    a = np.asarray(a).ravel().astype(np.uint8)
    if m is None:
        m = int(round(np.sqrt(a.size)))
    par, _, _, _ = _uf_components(a, m)
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    size = {}
    for i in range(m * m):
        if a[i]:
            r = find(i); size[r] = size.get(r, 0) + 1
    out = a.copy()
    for i in range(m * m):
        if a[i] and size[find(i)] < min_size:
            out[i] = 0
    O = out.reshape(m, m)
    filled = O.copy()
    for y in range(m):
        for x in range(m):
            if O[y, x]:
                continue
            l = O[y, (x - 1) % m]; r = O[y, (x + 1) % m]
            u = O[(y - 1) % m, x]; d = O[(y + 1) % m, x]
            if l and r and u and d:
                filled[y, x] = 1
    return filled.ravel().astype(np.uint8)

# ------------------------------- SIMP inverse-homogenization polish (port)
_FILT_CACHE = {}

def _filter_matrix(m, rmin):
    """Cone density filter as a sparse row-normalized matrix H: filt = H @ x."""
    key = (m, rmin)
    if key in _FILT_CACHE:
        return _FILT_CACHE[key]
    rr = int(np.floor(rmin))
    rows = []; cols = []; vals = []
    for ey in range(m):
        for ex in range(m):
            e = ey * m + ex; ws = 0.0; nb = []; w = []
            for dy in range(-rr, rr + 1):
                for dx in range(-rr, rr + 1):
                    dist = np.hypot(dx, dy)
                    if dist >= rmin:
                        continue
                    j = ((ey + dy) % m) * m + ((ex + dx) % m)
                    nb.append(j); w.append(rmin - dist); ws += rmin - dist
            for j, wv in zip(nb, w):
                rows.append(e); cols.append(j); vals.append(wv / ws)
    H = sparse.csr_matrix((vals, (rows, cols)), shape=(m * m, m * m))
    _FILT_CACHE[key] = H
    return H

def optimize_cell(solid, m, target_vf, iters=15, p=3, rmin=2.0, move=0.2,
                  Emin=1e-3, objective="bulk", beta=0.8, symmetric=False,
                  record=True, init_density=None):
    """SIMP polish. Returns list of binary cells after each iteration (len=iters)
    when record=True (the correction-budget trajectory), else the final cell.
    init_density (nel,) overrides the default 0.9/0.1-from-binary start — used for
    cold-start controls (e.g. a uniform field)."""
    nel = m * m
    edof = _edof(m)
    H = _filter_matrix(m, rmin); HT = H.T.tocsr()
    def filt(field): return H.dot(field)
    def filtT(dphys): return HT.dot(dphys)

    def project(xphys):
        srt = np.sort(xphys)
        kth = int(np.floor((1 - target_vf) * nel))
        thr = srt[min(nel - 1, max(0, kth))]
        return (xphys >= thr).astype(np.uint8)

    if init_density is not None:
        x = np.clip(np.asarray(init_density).ravel().astype(float), 0.01, 0.99)
    else:
        a0 = np.asarray(solid).ravel()
        x = np.where(a0 > 0, 0.9, 0.1).astype(float)
    xphys = filt(x)
    dv = filtT(np.full(nel, 1.0 / nel))
    traj = []
    for _ in range(iters):
        scale = Emin + (1 - Emin) * xphys ** p
        dscale = (1 - Emin) * p * xphys ** (p - 1)
        chi = _solve_chi(scale.astype(float), m, edof)
        E00, E11, E01 = _element_energies(chi, m, edof)
        V = nel
        if objective == "auxetic":
            dObj = dscale * (E01 - beta * (E00 + E11)) / V
        else:
            dObj = -dscale * (E00 + E11 + 2 * E01) / 4.0 / V
        dc = filtT(dObj)
        l1, l2 = 1e-9, 1e9
        xn = x.copy()
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lm = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dc / (dv * lm + 1e-30))
            v = x * np.sqrt(be)
            v = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, v))))
            xn = v
            vol = float(np.mean(filt(xn)))
            if vol > target_vf:
                l1 = lm
            else:
                l2 = lm
        if symmetric:
            X = xn.reshape(m, m)
            xn = (0.25 * (X + X[:, ::-1] + X[::-1, :] + X[::-1, ::-1])).ravel()
        x = xn
        xphys = filt(x)
        if record:
            traj.append(project(xphys))
    if record:
        return traj
    return project(xphys)

# ------------------------------- proper cold-start topology optimization
def full_to(m, target_vf, iters=80, rmin=2.0, move=0.2, Emin=1e-3, seed=0,
            record_every=4):
    """Legitimate 'full TO from scratch' for max bulk modulus: uniform init with a
    deterministic symmetry-breaking seed, penalization CONTINUATION (p: 1->4), OC
    update, density filter, projected once at the end. This is what the lightweight
    polish operator can NOT do from cold. Returns (final_binary, traj) where traj is
    a list of (iter, binary) snapshots and the continuous bulk at each iter."""
    nel = m * m
    edof = _edof(m)
    H = _filter_matrix(m, rmin); HT = H.T.tocsr()
    filt = lambda f: H.dot(f); filtT = lambda d: HT.dot(d)
    dv = filtT(np.full(nel, 1.0 / nel))
    # uniform + small deterministic symmetry-breaking pattern (reproducible)
    yy, xx = np.mgrid[0:m, 0:m]
    seed_field = 0.06 * (np.sin(2 * np.pi * xx / m) * np.sin(2 * np.pi * yy / m)
                         + 0.5 * np.cos(4 * np.pi * (xx + yy) / m))
    x = np.clip(target_vf + seed_field.ravel(), 0.05, 0.95)
    def project(xphys):
        srt = np.sort(xphys); kth = int(np.floor((1 - target_vf) * nel))
        thr = srt[min(nel - 1, max(0, kth))]
        return (xphys >= thr).astype(np.uint8)
    traj = []; bulk_hist = []
    for it in range(iters):
        p = 1.0 + 3.0 * it / max(1, iters - 1)              # continuation 1 -> 4
        xphys = filt(x)
        scale = Emin + (1 - Emin) * xphys ** p
        dscale = (1 - Emin) * p * xphys ** (p - 1)
        chi = _solve_chi(scale.astype(float), m, edof)
        E00, E11, E01 = _element_energies(chi, m, edof)
        V = nel
        bulk_hist.append((np.sum(scale * E00) + np.sum(scale * E11) + 2 * np.sum(scale * E01)) / 4.0 / V)
        dObj = -dscale * (E00 + E11 + 2 * E01) / 4.0 / V
        dc = filtT(dObj)
        l1, l2 = 1e-9, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lm = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dc / (dv * lm + 1e-30))
            v = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(be)))))
            if float(np.mean(filt(v))) > target_vf: l1 = lm
            else: l2 = lm
        x = v
        if record_every and (it % record_every == 0 or it == iters - 1):
            traj.append((it + 1, project(filt(x))))
    return project(filt(x)), traj, bulk_hist


# ------------------------------- guidance gradient (port of stiffnessGrad)
def stiffness_grad(dens, m, p=3, Emin=1e-3, objective="bulk", beta=0.8):
    """Ascent gradient of the objective w.r.t. density for one continuous cell."""
    nel = m * m
    edof = _edof(m)
    d = np.clip(np.asarray(dens).ravel(), 0, 1)
    scale = Emin + (1 - Emin) * d ** p
    dscale = (1 - Emin) * p * d ** (p - 1)
    chi = _solve_chi(scale.astype(float), m, edof)
    E00, E11, E01 = _element_energies(chi, m, edof)
    V = nel
    C11 = float(np.sum(scale * E11) / V); C01 = float(np.sum(scale * E01) / V)
    if objective == "auxetic":
        g = -dscale * (E01 - beta * (E00 + E11)) / V
    else:
        g = dscale * (E00 + E11 + 2 * E01) / 4.0 / V
    return g, C01 / (C11 + 1e-12)


if __name__ == "__main__":
    # 1. analytic checks
    solid = np.ones(32 * 32, np.uint8)
    h = homogenize(solid, 32)
    print("solid:  E=%.4f nu=%.4f (expect ~1.0, 0.30)" % (h["E"], h["nu"]))
    lay = np.zeros((32, 32), np.uint8); lay[::2, :] = 1
    h = homogenize(lay, 32)
    print("layered: Ex=%.3f Ey=%.3f (Voigt/Reuss-like)" % (h["Ex"], h["Ey"]))
    # 2. finite-difference check of the guidance gradient (expect small rel err)
    rng = np.random.default_rng(0)
    dens = rng.random(32 * 32) * 0.6 + 0.2
    g, _ = stiffness_grad(dens, 32)
    def bulk_of(dd):
        sc = 1e-3 + (1 - 1e-3) * np.clip(dd, 0, 1) ** 3
        edof = _edof(32); chi = _solve_chi(sc, 32, edof)
        E00, E11, E01 = _element_energies(chi, 32, edof); V = 32 * 32
        return (np.sum(sc * E00) + np.sum(sc * E11) + 2 * np.sum(sc * E01)) / 4.0 / V
    idx = [10, 200, 512, 900]; eps = 1e-4; errs = []
    for i in idx:
        dp = dens.copy(); dp[i] += eps; dm = dens.copy(); dm[i] -= eps
        fd = (bulk_of(dp) - bulk_of(dm)) / (2 * eps)
        errs.append(abs(fd - g[i]) / (abs(fd) + 1e-12))
    print("guidance grad vs finite diff, max rel err: %.2e" % max(errs))
