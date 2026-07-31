"""
Auxetic specialist DDPM: trained ONLY on the 2000-cell rotating-squares
dataset, conditioned on [volume_fraction, nu] (2 dims, both scalar).

Why a dedicated model: four attempts showed the small mixed-model UNet cannot
switch geometry FAMILY from a conditioning channel (warm-started scalar nu,
from-scratch scalar nu, spatial cond planes, and an aux-class one-hot all
failed to produce rotating squares on demand; the channel only biases
statistics). Within a homogeneous family, conditioning is a smooth
geometry-parameter mapping - the same kind of scalar control v2 demonstrably
learned for vf - so the specialist handles nu, and the proven v2 model keeps
handling non-auxetic requests. The demo routes between the two by target nu.

Usage:
  python ddpm_aux.py train --epochs 80 --snapshot-every 10
  python ddpm_aux.py probe --ckpt checkpoints/ddpmaux_ep60.pt
"""
import os, csv, math, copy, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG = 32
T_STEPS = 400
COND_DIM = 2
NU_CLIP = 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(LAB, "Dataset Preparation", "auxetic_dataset")
CKPT = os.path.join(HERE, "checkpoints", "ddpmaux.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

betas = torch.linspace(1e-4, 0.02, T_STEPS)
abar = torch.cumprod(1 - betas, dim=0)


class Cells(Dataset):
    def __init__(self):
        rows = list(csv.DictReader(open(os.path.join(DATA, "auxetic_metrics.csv"))))
        self.x = torch.empty(len(rows), 1, IMG, IMG)
        self.c = torch.empty(len(rows), COND_DIM)
        for i, r in enumerate(rows):
            img = Image.open(os.path.join(DATA, "images", r["filename"])) \
                .convert("L").resize((IMG, IMG), Image.BILINEAR)
            a = (np.array(img, np.float32) / 255.0 > 0.5).astype(np.float32) * 2 - 1
            self.x[i, 0] = torch.from_numpy(a)
            nu = float(np.clip(float(r["poisson_eff"]), -NU_CLIP, NU_CLIP))
            self.c[i] = torch.tensor([float(r["volume_fraction"]), nu])

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


def q_sample(x0, t, n):
    a = abar.to(x0.device)[t][:, None, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * n


def train(epochs, snapshot_every=10, p_drop=0.12, lr=2e-4, batch=128,
          ema_decay=0.999):
    torch.manual_seed(0)
    ds = Cells(); dl = DataLoader(ds, batch_size=batch, shuffle=True)
    print(f"dataset: {len(ds)} auxetic cells")
    model = UNet().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = copy.deepcopy(model); [p.requires_grad_(False) for p in ema.parameters()]
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"]); start = ck["epoch"]
        print(f"resumed at epoch {start}")
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
        if ep % 5 == 0 or ep == start + 1:
            print(f"epoch {ep:3d} loss {tot/len(dl):.4f}", flush=True)
        state = {"state": model.state_dict(), "ema": ema.state_dict(),
                 "opt": opt.state_dict(), "epoch": ep}
        torch.save(state, CKPT)
        if ep % snapshot_every == 0:
            torch.save(state, CKPT.replace(".pt", f"_ep{ep:02d}.pt"))
    print(f"saved at epoch {start + epochs}")


def make_cond(vf, nu):
    return [float(vf), float(np.clip(nu, -NU_CLIP, NU_CLIP))]


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


def probe(ckpt_path, n=16, guidance=3.0, seed=0):
    from metrics import homogenize, connectivity
    m = load_model(ckpt_path)
    print(f"probe of {os.path.basename(ckpt_path)} (n={n}, guidance {guidance})")
    for vf_req, nu_req in ((0.45, -0.8), (0.45, -0.5), (0.45, -0.2), (0.45, 0.0),
                           (0.38, -0.5), (0.55, -0.5)):
        cells = ddim_sample(m, n, make_cond(vf_req, nu_req),
                            guidance=guidance, seed=seed)
        nus, con = [], 0
        for c in cells:
            info = connectivity(c)
            if info["periodic_connected_x"] and info["periodic_connected_y"]:
                con += 1
            try:
                nus.append(homogenize(c)["poisson_eff"])
            except Exception:
                pass
        print(f"  req vf {vf_req:.2f} nu {nu_req:+.2f} -> "
              f"vf {np.mean([c.mean() for c in cells]):.3f} "
              f"nu~{np.median(nus):+.3f}  conn {con/n:.0%}  "
              f"aux {sum(1 for v in nus if v < -0.01)}/{len(nus)}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--snapshot-every", type=int, default=10)
    p = sub.add_parser("probe")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("-n", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "train":
        train(epochs=args.epochs, snapshot_every=args.snapshot_every)
    else:
        probe(args.ckpt, n=args.n, guidance=args.guidance)


if __name__ == "__main__":
    main()
