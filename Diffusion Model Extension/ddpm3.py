"""
Diffusion v3: adds Poisson's-ratio (nu) conditioning for auxetic generation.

Condition = [volume_fraction (1)]
          + [class one-hot: low/med/high stiffness, aux-mild, aux-strong (5)]
          = 6 dims.
Cells with measured nu < -0.05 are classed auxetic (aux-strong if nu < -0.35,
else aux-mild), replacing their stiffness class; all other cells keep the v2
stiffness terciles. This reuses the DISCRETE class mechanism that v2
demonstrably learns with this architecture and recipe. Three failed attempts
taught us what does NOT work here: (a) warm-started fine-tuning never grows a
new cond channel, (b) a scalar nu channel through the embedding MLP stays
inert even from scratch, (c) TopoDiff-style spatial cond planes need far more
training than this budget. Fine nu targeting within a class is handled by the
project's standard generate-then-filter step. Auxetic cells oversampled ~3x.

Data = the 8040-cell Stage-3 dataset + ~2000 rotating-squares auxetic cells
(make_auxetic_dataset.py), both labeled by the same Stage-3 homogenizer.

Checkpoint selection is by CONTROLLABILITY, not loss (v2 lesson: loss-based
overtraining weakens conditioning). `train` snapshots periodically; `probe`
measures vf tracking and nu targeting of a snapshot with the FEM evaluator.

Usage:
  python ddpm3.py train --epochs 30 --snapshot-every 3
  python ddpm3.py probe --ckpt checkpoints/ddpm3_ep09.pt
  python ddpm3.py sample --vf 0.45 --stiff medium --nu -0.4 -n 16
"""
import os, sys, csv, math, copy, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG = 32
T_STEPS = 400
COND_DIM = 6
NU_CLIP = 0.8
AUX_THRESH = -0.05
AUX_STRONG = -0.35
CLASSES = ["low", "medium", "high", "aux_mild", "aux_strong"]
STIFF = ["low", "medium", "high"]
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.normpath(os.path.join(HERE, ".."))
DATA_MAIN = os.path.join(LAB, "Dataset Preparation", "training_dataset_8040", "images")
CSV_MAIN = os.path.join(LAB, "Simple Mechanical Analysis", "stage3_metrics_8040.csv")
DATA_AUX = os.path.join(LAB, "Dataset Preparation", "auxetic_dataset", "images")
CSV_AUX = os.path.join(LAB, "Dataset Preparation", "auxetic_dataset", "auxetic_metrics.csv")
CKPT_V2 = os.path.join(HERE, "checkpoints", "ddpm2_ep25.pt")
CKPT = os.path.join(HERE, "checkpoints", "ddpm3.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

betas = torch.linspace(1e-4, 0.02, T_STEPS)
abar = torch.cumprod(1 - betas, dim=0)


def stiff_onehot(e, edges):
    i = 0 if e < edges[0] else (1 if e < edges[1] else 2)
    v = [0., 0., 0.]; v[i] = 1.; return v


def load_rows():
    rows = []
    for csv_path, img_dir in ((CSV_MAIN, DATA_MAIN), (CSV_AUX, DATA_AUX)):
        for r in csv.DictReader(open(csv_path)):
            e = 0.5 * (float(r["E_eff_x_rel"]) + float(r["E_eff_y_rel"]))
            nu = float(np.clip(float(r["poisson_eff"]), -NU_CLIP, NU_CLIP))
            rows.append((os.path.join(img_dir, r["filename"]),
                         float(r["volume_fraction"]), e, nu))
    return rows


class Cells(Dataset):
    """Preloads every cell into memory (10k x 32x32 floats = ~40 MB)."""
    def __init__(self):
        rows = load_rows()
        E = np.array([r[2] for r in rows])
        self.edges = (float(np.quantile(E, 1/3)), float(np.quantile(E, 2/3)))
        self.x = torch.empty(len(rows), 1, IMG, IMG)
        self.c = torch.empty(len(rows), COND_DIM)
        self.w = torch.empty(len(rows))          # sampling weights
        for i, (path, vf, e, nu) in enumerate(rows):
            img = Image.open(path).convert("L").resize((IMG, IMG), Image.BILINEAR)
            a = (np.array(img, np.float32) / 255.0 > 0.5).astype(np.float32) * 2 - 1
            self.x[i, 0] = torch.from_numpy(a)
            if nu < AUX_STRONG:
                ci = 4
            elif nu < AUX_THRESH:
                ci = 3
            else:
                ci = stiff_onehot(e, self.edges).index(1.0)
            onehot = [0.] * 5; onehot[ci] = 1.
            self.c[i] = torch.tensor([vf] + onehot)
            self.w[i] = 3.0 if ci >= 3 else 1.0

    def __len__(self): return self.x.size(0)

    def __getitem__(self, i): return self.x[i], self.c[i]


def time_embed(t, dim):
    half = dim // 2
    fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * fr[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ResBlock(nn.Module):
    def __init__(self, ci, co, ed):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.emb = nn.Linear(ed, co)
        self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.sk(x)


class UNet(nn.Module):
    def __init__(self, ed=128):
        super().__init__(); self.ed = ed
        self.t_mlp = nn.Sequential(nn.Linear(ed, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.c_mlp = nn.Sequential(nn.Linear(COND_DIM, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.null = nn.Parameter(torch.zeros(ed))
        self.inc = nn.Conv2d(1, 32, 3, 1, 1)
        self.d1 = ResBlock(32, 32, ed); self.p1 = nn.Conv2d(32, 32, 4, 2, 1)
        self.d2 = ResBlock(32, 64, ed); self.p2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.d3 = ResBlock(64, 128, ed); self.p3 = nn.Conv2d(128, 128, 4, 2, 1)
        self.m1 = ResBlock(128, 128, ed); self.m2 = ResBlock(128, 128, ed)
        self.u3 = nn.ConvTranspose2d(128, 128, 4, 2, 1); self.r3 = ResBlock(256, 64, ed)
        self.u2 = nn.ConvTranspose2d(64, 64, 4, 2, 1); self.r2 = ResBlock(128, 32, ed)
        self.u1 = nn.ConvTranspose2d(32, 32, 4, 2, 1); self.r1 = ResBlock(64, 32, ed)
        self.out = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, x, t, cond, drop=None):
        e = self.t_mlp(time_embed(t, self.ed))
        ce = self.c_mlp(cond)
        if drop is not None:
            ce = torch.where(drop[:, None], self.null.expand_as(ce), ce)
        e = e + ce
        h0 = self.inc(x); h1 = self.d1(h0, e)
        h2 = self.d2(self.p1(h1), e); h3 = self.d3(self.p2(h2), e)
        m = self.m2(self.m1(self.p3(h3), e), e)
        u = self.r3(torch.cat([self.u3(m), h3], 1), e)
        u = self.r2(torch.cat([self.u2(u), h2], 1), e)
        u = self.r1(torch.cat([self.u1(u), h1], 1), e)
        return self.out(u)

    def forward_uncond(self, x, t):
        b = x.size(0)
        return self.forward(x, t, torch.zeros(b, COND_DIM, device=x.device),
                            torch.ones(b, dtype=torch.bool, device=x.device))


def widen_cond_state(sd):
    """Map a narrower state dict to COND_DIM: zero-init the new columns."""
    w = sd["c_mlp.0.weight"]
    if w.shape[1] == COND_DIM:
        return sd
    assert w.shape[1] < COND_DIM, f"unexpected c_mlp input dim {w.shape[1]}"
    w2 = torch.zeros(w.shape[0], COND_DIM)
    w2[:, :w.shape[1]] = w
    sd = dict(sd)
    sd["c_mlp.0.weight"] = w2
    return sd


def q_sample(x0, t, n):
    a = abar.to(x0.device)[t][:, None, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * n


def train(epochs, snapshot_every=3, p_drop=0.12, lr=2e-4, batch=128,
          ema_decay=0.999, scratch=False):
    torch.manual_seed(0)
    ds = Cells()
    sampler = torch.utils.data.WeightedRandomSampler(
        ds.w, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=batch, sampler=sampler)
    print(f"dataset: {len(ds)} cells, stiffness edges {ds.edges}")
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = copy.deepcopy(model); [p.requires_grad_(False) for p in ema.parameters()]
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"]); start = ck["epoch"]
        print(f"resumed ddpm3 at epoch {start}")
    else:
        # v2 lesson confirmed the hard way: fine-tuning a converged model does
        # not grow NEW conditioning channels (nu stayed dead through 30-40
        # epochs of warm-started training). Conditioning must co-learn with
        # the network, so v3 trains from scratch like v2 did.
        print("training from scratch (random init)")
    model.train()
    for ep in range(start + 1, start + epochs + 1):
        tot = 0.0
        for x, c in dl:
            x, c = x.to(DEVICE), c.to(DEVICE); b = x.size(0)
            t = torch.randint(0, T_STEPS, (b,), device=DEVICE)
            n = torch.randn_like(x)
            drop = torch.rand(b, device=DEVICE) < p_drop
            pred = model(q_sample(x, t, n), t, c, drop)
            loss = F.mse_loss(pred, n)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)
        print(f"epoch {ep:3d} loss {tot/len(dl):.4f}", flush=True)
        state = {"state": model.state_dict(), "ema": ema.state_dict(),
                 "opt": opt.state_dict(), "epoch": ep, "edges": ds.edges}
        torch.save(state, CKPT)
        if ep % snapshot_every == 0:
            torch.save(state, CKPT.replace(".pt", f"_ep{ep:02d}.pt"))
    print(f"saved at epoch {start + epochs}")


def make_cond(vf, cls, nu=None):
    """cls is one of CLASSES; nu kept for backward-compat, maps to a class."""
    if nu is not None and nu < AUX_THRESH:
        cls = "aux_strong" if nu < AUX_STRONG else "aux_mild"
    v = [float(vf), 0., 0., 0., 0., 0.]
    v[1 + CLASSES.index(cls)] = 1.
    return v


@torch.no_grad()
def ddim_sample(model, n, cond_vec, steps=50, guidance=3.0, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model.eval(); ab = abar.to(DEVICE)
    ts = torch.linspace(T_STEPS - 1, 0, steps, dtype=torch.long, device=DEVICE)
    x = torch.randn(n, 1, IMG, IMG, device=DEVICE)
    c = torch.tensor(cond_vec, dtype=torch.float32, device=DEVICE).expand(n, COND_DIM)
    for i in range(steps):
        t = ts[i].expand(n)
        ec = model(x, t, c); eu = model.forward_uncond(x, t)
        eps = eu + guidance * (ec - eu)
        a_t = ab[ts[i]]
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = (ab[ts[i+1]].sqrt() * x0 + (1 - ab[ts[i+1]]).sqrt() * eps) if i < steps-1 else x0
    p = (x.cpu().numpy()[:, 0] + 1) / 2
    return (p >= 0.5).astype(np.uint8)


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = UNet().to(DEVICE); m.load_state_dict(ck["ema"])
    return m


def probe(ckpt_path, n=24, guidance=3.0, seed=0):
    """Controllability probe: vf tracking + nu targeting, FEM-measured."""
    from metrics import homogenize, connectivity
    m = load_model(ckpt_path)
    print(f"probe of {os.path.basename(ckpt_path)} (n={n}, guidance {guidance})")

    def measure(cells):
        vfs, nus, con = [], [], 0
        for c in cells:
            vfs.append(c.mean())
            info = connectivity(c)
            if info["periodic_connected_x"] and info["periodic_connected_y"]:
                con += 1
            try:
                h = homogenize(c)
                nus.append(h["poisson_eff"])
            except Exception:
                pass
        return (float(np.mean(vfs)),
                float(np.median(nus)) if nus else float("nan"),
                con / len(cells))

    print("vf tracking (stiff medium, nu 0.3):")
    for vf_req in (0.3, 0.45, 0.6):
        cells = ddim_sample(m, n, make_cond(vf_req, "medium", 0.3),
                            guidance=guidance, seed=seed)
        vf, nu, con = measure(cells)
        print(f"  req vf {vf_req:.2f} -> vf {vf:.3f}  nu~{nu:.3f}  conn {con:.0%}")
    print("nu targeting (vf 0.45, stiff medium):")
    for nu_req in (-0.6, -0.3, -0.1, 0.3):
        cells = ddim_sample(m, n, make_cond(0.45, "medium", nu_req),
                            guidance=guidance, seed=seed)
        vf, nu, con = measure(cells)
        print(f"  req nu {nu_req:+.2f} -> nu~{nu:+.3f}  vf {vf:.3f}  conn {con:.0%}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--snapshot-every", type=int, default=3)
    t.add_argument("--scratch", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("-n", type=int, default=24)
    s = sub.add_parser("sample")
    s.add_argument("--vf", type=float, default=0.45)
    s.add_argument("--stiff", default="medium", choices=STIFF)
    s.add_argument("--nu", type=float, default=-0.3)
    s.add_argument("-n", type=int, default=16)
    s.add_argument("--guidance", type=float, default=3.0)
    s.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    if args.cmd == "train":
        train(epochs=args.epochs, snapshot_every=args.snapshot_every,
              scratch=args.scratch)
    elif args.cmd == "probe":
        probe(args.ckpt, n=args.n, guidance=args.guidance)
    else:
        m = load_model(args.ckpt)
        cells = ddim_sample(m, args.n, make_cond(args.vf, args.stiff, args.nu),
                            guidance=args.guidance, seed=0)
        print("mean vf:", round(float(np.mean([c.mean() for c in cells])), 3))


if __name__ == "__main__":
    main()
