"""
Baseline generative model for 2D unit cells.

A convolutional VAE with an optional conditioning vector. Train it
unconditional first (sanity baseline), then conditional on:
  - volume fraction (continuous, 1 value)
  - stiffness class (low/medium/high from homogenized E, one-hot 3)
  - symmetry class (square/biaxial/uniaxial/diagonal/none, one-hot 5)

Conditioning is concatenated to the encoder input as constant channels and to
the latent code at the decoder, the standard CVAE recipe.

Labels come from unitcells/metrics.csv (the Stage 3 homogenized labels), NOT
the Stage 2 heuristics, so the generator is conditioned on the same quantities
the evaluator measures.

Usage:
  python cvae.py train --epochs 150            # trains both models
  python cvae.py sample --vf 0.3 --stiff low --sym square -n 16
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG = 64
LATENT = 32
STIFF_CLASSES = ["low", "medium", "high"]
SYM_CLASSES = ["square", "biaxial", "uniaxial", "diagonal", "none"]
COND_DIM = 1 + len(STIFF_CLASSES) + len(SYM_CLASSES)   # 9
DATA_DIR = "/home/claude/unitcells"
CKPT_DIR = "/home/claude/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# stiffness-class bin edges on mean relative modulus (set from data terciles
# at training time and stored in the checkpoint so sampling uses the same bins)
DEFAULT_EDGES = (0.05, 0.20)


def stiffness_class(e_mean, edges):
    if e_mean < edges[0]:
        return "low"
    if e_mean < edges[1]:
        return "medium"
    return "high"


def make_cond(vf, stiff, sym):
    v = np.zeros(COND_DIM, dtype=np.float32)
    v[0] = vf
    v[1 + STIFF_CLASSES.index(stiff)] = 1.0
    v[1 + len(STIFF_CLASSES) + SYM_CLASSES.index(sym)] = 1.0
    return v


class CellDataset(Dataset):
    def __init__(self, img_dir, metrics_csv):
        rows = list(csv.DictReader(open(metrics_csv)))
        # tercile edges on mean modulus so classes are balanced
        e = np.array([(float(r["E_eff_x_rel"]) + float(r["E_eff_y_rel"])) / 2
                      for r in rows])
        self.edges = (float(np.quantile(e, 1 / 3)), float(np.quantile(e, 2 / 3)))
        self.items = []
        for r in rows:
            em = (float(r["E_eff_x_rel"]) + float(r["E_eff_y_rel"])) / 2
            cond = make_cond(float(r["volume_fraction"]),
                             stiffness_class(em, self.edges),
                             r["symmetry_class"])
            self.items.append((os.path.join(img_dir, r["filename"]), cond))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cond = self.items[i]
        x = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(x)[None], torch.from_numpy(cond)


class CVAE(nn.Module):
    def __init__(self, latent=LATENT, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        cin = 1 + cond_dim          # condition tiled as constant image channels
        self.enc = nn.Sequential(
            nn.Conv2d(cin, 32, 4, 2, 1), nn.ReLU(),    # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),     # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),    # 8x8
            nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU(),   # 4x4
            nn.Flatten())
        self.fc_mu = nn.Linear(128 * 16, latent)
        self.fc_lv = nn.Linear(128 * 16, latent)
        self.fc_up = nn.Linear(latent + cond_dim, 128 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(),  # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 32x32
            nn.ConvTranspose2d(32, 1, 4, 2, 1))                # 64x64 logits

    def encode(self, x, c):
        if self.cond_dim:
            cm = c[:, :, None, None].expand(-1, -1, IMG, IMG)
            x = torch.cat([x, cm], dim=1)
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z, c):
        if self.cond_dim:
            z = torch.cat([z, c], dim=1)
        h = self.fc_up(z).view(-1, 128, 4, 4)
        return self.dec(h)

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, c), mu, lv


def loss_fn(logits, x, mu, lv, beta):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return rec + beta * kl, rec.item(), kl.item()


def train(epochs=150, beta=1.0, lr=1e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ds = CellDataset(os.path.join(DATA_DIR, "images"),
                     os.path.join(DATA_DIR, "metrics.csv"))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    print(f"dataset: {len(ds)} cells, stiffness bin edges {ds.edges}")

    for cond_dim, name in [(0, "vae"), (COND_DIM, "cvae")]:
        model = CVAE(cond_dim=cond_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            model.train()
            tot = trec = tkl = 0.0
            # KL warmup over the first third of training
            b = beta * min(1.0, ep / max(1, epochs // 3))
            for x, c in dl:
                x, c = x.to(DEVICE), c.to(DEVICE)
                logits, mu, lv = model(x, c)
                loss, rec, kl = loss_fn(logits, x, mu, lv, b)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); trec += rec; tkl += kl
            n = len(dl)
            if ep % 25 == 0 or ep == 1:
                print(f"[{name}] epoch {ep:3d} loss {tot/n:8.1f} "
                      f"rec {trec/n:8.1f} kl {tkl/n:6.1f}")
        torch.save({"state": model.state_dict(), "cond_dim": cond_dim,
                    "edges": ds.edges}, os.path.join(CKPT_DIR, f"{name}.pt"))
        print(f"saved {name}.pt")


def load(name):
    ck = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"),
                    map_location=DEVICE, weights_only=False)
    m = CVAE(cond_dim=ck["cond_dim"]).to(DEVICE)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck


@torch.no_grad()
def sample(model, n, cond=None, threshold=0.5, symmetrize=False):
    z = torch.randn(n, LATENT, device=DEVICE)
    c = (torch.from_numpy(np.tile(cond, (n, 1))).to(DEVICE)
         if cond is not None else torch.zeros(n, 0, device=DEVICE))
    probs = torch.sigmoid(model.decode(z, c)).cpu().numpy()[:, 0]
    if symmetrize:
        # average the four mirror images of the probability field, then
        # threshold: guarantees exact biaxial symmetry (DiffMat-style quarter
        # design) and cleans ragged edges
        probs = 0.25 * (probs + probs[:, :, ::-1] + probs[:, ::-1, :]
                        + probs[:, ::-1, ::-1])
        if symmetrize == "square":           # also fold across the diagonal
            probs = 0.5 * (probs + np.transpose(probs, (0, 2, 1)))
    return (probs >= threshold).astype(np.uint8)


def save_grid(cells, path, cols=8, pad=3):
    n = len(cells); rows = (n + cols - 1) // cols
    W = cols * IMG + (cols + 1) * pad
    H = rows * IMG + (rows + 1) * pad
    canvas = np.full((H, W), 180, np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        y = pad + r * (IMG + pad); x = pad + k * (IMG + pad)
        canvas[y:y + IMG, x:x + IMG] = c * 255
    Image.fromarray(canvas).save(path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--beta", type=float, default=1.0)
    s = sub.add_parser("sample")
    s.add_argument("--model", default="cvae", choices=["vae", "cvae"])
    s.add_argument("--vf", type=float, default=0.4)
    s.add_argument("--stiff", default="medium", choices=STIFF_CLASSES)
    s.add_argument("--sym", default="biaxial", choices=SYM_CLASSES)
    s.add_argument("-n", type=int, default=16)
    s.add_argument("--symmetrize", action="store_true",
                   help="enforce biaxial symmetry by mirror-averaging")
    s.add_argument("--out", default="/home/claude/samples")
    args = ap.parse_args()

    if args.cmd == "train":
        train(epochs=args.epochs, beta=args.beta)
    else:
        model, ck = load(args.model)
        cond = (make_cond(args.vf, args.stiff, args.sym)
                if ck["cond_dim"] else None)
        cells = sample(model, args.n, cond, symmetrize=args.symmetrize)
        os.makedirs(args.out, exist_ok=True)
        tag = (f"{args.model}_vf{args.vf}_{args.stiff}_{args.sym}"
               if cond is not None else args.model)
        for i, c in enumerate(cells):
            Image.fromarray(c * 255).save(
                os.path.join(args.out, f"{tag}_{i:03d}.png"))
        save_grid(cells, os.path.join(args.out, f"{tag}_grid.png"))
        print(f"wrote {args.n} samples + grid to {args.out}")


if __name__ == "__main__":
    main()
