"""Chunked trainer: resumes from checkpoint so each call stays short.
Usage: python train_chunk.py <vae|cvae> <epochs>"""
import os, sys, torch
from torch.utils.data import DataLoader
from cvae import (CVAE, CellDataset, loss_fn, COND_DIM, DATA_DIR, CKPT_DIR,
                  DEVICE)

name = sys.argv[1]
epochs = int(sys.argv[2])
cond_dim = COND_DIM if name == "cvae" else 0
torch.manual_seed(0)
os.makedirs(CKPT_DIR, exist_ok=True)

ds = CellDataset(os.path.join(DATA_DIR, "images"),
                 os.path.join(DATA_DIR, "metrics.csv"))
dl = DataLoader(ds, batch_size=64, shuffle=True)

model = CVAE(cond_dim=cond_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
start_ep = 0
path = os.path.join(CKPT_DIR, f"{name}.pt")
if os.path.exists(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    if ck.get("epoch"):                      # only resume chunked checkpoints
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"]

TOTAL = 100                                   # for KL warmup schedule
for ep in range(start_ep + 1, start_ep + epochs + 1):
    model.train()
    tot = 0.0
    b = min(1.0, ep / (TOTAL // 3))
    for x, c in dl:
        x, c = x.to(DEVICE), c.to(DEVICE)
        logits, mu, lv = model(x, c)
        loss, rec, kl = loss_fn(logits, x, mu, lv, b)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if ep % 10 == 0:
        print(f"[{name}] epoch {ep:3d} loss {tot/len(dl):8.1f}")

torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
            "cond_dim": cond_dim, "edges": ds.edges, "epoch": start_ep + epochs},
           path)
print(f"saved {name}.pt at epoch {start_ep + epochs}")
