"""
Label a folder of unit-cell images with the two quantities used to condition the
diffusion model: volume fraction (exact) and effective Young's modulus (fast
conjugate-gradient homogenization at 24x24, relative to the solid material).

Writes <folder>/labels_cond.csv with columns: filename, volume_fraction, E_mean_rel.
Resumable: re-running appends only the not-yet-labeled images, in chunks.

Usage:
  python label_dataset.py <folder> [chunk_size]
  # e.g. python label_dataset.py unitcells_big 1500   (run repeatedly to finish)

For richer labels (connectivity, symmetry, full elastic tensor, Poisson), use
metrics.py instead; this script is the lightweight labeler for conditioning.
"""
import os, csv, sys, glob
import numpy as np
from PIL import Image

if len(sys.argv) < 2:
    print("usage: python label_dataset.py <folder> [chunk_size]"); sys.exit(1)
FOLDER = sys.argv[1]
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
IMG_DIR = os.path.join(FOLDER, "images")
OUT = os.path.join(FOLDER, "labels_cond.csv")


def ke_chi0(nu0=0.3):
    a = b = 0.5
    C = (1 / (1 - nu0**2)) * np.array([[1, nu0, 0], [nu0, 1, 0], [0, 0, (1 - nu0) / 2]])
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    ke = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / a
            dNy = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / b
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx; B[1, 1::2] = dNy; B[2, 0::2] = dNy; B[2, 1::2] = dNx
            ke += (B.T @ C @ B) * (a * b)
    chi0 = np.array([[-a, 0, a, 0, a, 0, -a, 0], [0, -b, 0, -b, 0, b, 0, b],
                     [-b, 0, -b, 0, b, 0, b, 0]]).T
    return ke, chi0
KE, CHI0 = ke_chi0()


def homog(solid, void=1e-3, iters=300, tol=1e-5):
    """Effective modulus (mean of x,y) via matrix-free CG homogenization."""
    m = solid.shape[0]; nel = m * m; ndof = 2 * m * m
    scale = np.where(solid.ravel() > 0, 1.0, void)
    ex = np.tile(np.arange(m), m); ey = np.repeat(np.arange(m), m)
    nid = lambda ix, iy: (iy % m) * m + (ix % m)
    nodes = np.stack([nid(ex, ey), nid(ex + 1, ey), nid(ex + 1, ey + 1), nid(ex, ey + 1)], 1)
    edof = np.empty((nel, 8), int); edof[:, 0::2] = 2 * nodes; edof[:, 1::2] = 2 * nodes + 1
    fe = KE @ CHI0
    F = np.zeros((ndof, 3))
    for c in range(3):
        np.add.at(F[:, c], edof.ravel(), (scale[:, None] * fe[:, c][None, :]).ravel())
    diag = np.zeros(ndof); ked = np.diag(KE)
    for k in range(8):
        np.add.at(diag, edof[:, k], scale * ked[k])

    def mv(x):
        xe = x[edof]; ye = (xe @ KE.T) * scale[:, None]; y = np.zeros(ndof)
        for k in range(8):
            np.add.at(y, edof[:, k], ye[:, k])
        return y
    free = np.ones(ndof, bool); free[0] = free[1] = False
    chi = np.zeros((ndof, 3))
    for c in range(3):
        b = F[:, c].copy(); b[~free] = 0; x = np.zeros(ndof)
        r = b - mv(x); r[~free] = 0
        z = r / diag; z[~free] = 0; p = z.copy(); rz = r @ z; bn = np.sqrt(b @ b + 1e-30)
        for _ in range(iters):
            Ap = mv(p); Ap[~free] = 0; al = rz / (p @ Ap + 1e-30)
            x += al * p; r -= al * Ap
            if np.sqrt(r @ r) < tol * bn:
                break
            z = r / diag; z[~free] = 0; rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        chi[:, c] = x
    V = nel; CH = np.zeros((3, 3))
    chi_e = chi[edof]; d = CHI0.T[None] - np.transpose(chi_e, (0, 2, 1))
    for i in range(3):
        dik = d[:, i, :] @ KE
        for j in range(3):
            CH[i, j] = np.sum(scale * np.einsum('ek,ek->e', dik, d[:, j, :])) / V
    CH = 0.5 * (CH + CH.T)
    try:
        S = np.linalg.inv(CH)
        return float(max(0.0, 0.5 * (1 / S[0, 0] + 1 / S[1, 1])))
    except Exception:
        return 0.0


def main():
    files = sorted(os.path.basename(f) for f in glob.glob(IMG_DIR + "/*.png"))
    if not files:
        print(f"no PNGs found in {IMG_DIR}"); return
    done = set()
    if os.path.exists(OUT):
        done = {r["filename"] for r in csv.DictReader(open(OUT))}
    todo = [f for f in files if f not in done][:CHUNK]
    mode = "a" if done else "w"
    with open(OUT, mode, newline="") as fh:
        w = csv.writer(fh)
        if not done:
            w.writerow(["filename", "volume_fraction", "E_mean_rel"])
        for f in todo:
            im = Image.open(os.path.join(IMG_DIR, f)).convert("L")
            a64 = (np.array(im) > 127).astype(np.uint8)
            vf = float(a64.mean())
            a24 = (np.array(im.resize((24, 24), Image.BILINEAR)) > 127).astype(np.uint8)
            E = homog(a24) if a24.sum() > 4 else 0.0
            w.writerow([f, round(vf, 4), round(E, 5)])
    print(f"labeled {len(todo)}; total {len(done) + len(todo)}/{len(files)} -> {OUT}")


if __name__ == "__main__":
    main()
