"""
Stage 5 final pipeline (v2 model): generate unit cells with a target volume
fraction and stiffness class.

Pipeline: conditional DDPM (ddpm2, epoch 25, EMA weights, selected by
controllability) -> request calibration (correct the model's vf bias) ->
optional symmetrization -> rejection by the Stage 3 evaluator (vf tolerance AND
periodic connectivity). Accepted cells satisfy the constraint by measurement.

Usage:
  python generate2.py --vf 0.45 --stiff high -n 16 [--symmetrize] [--tol 0.05]
"""
import os, argparse
import numpy as np
import torch
from PIL import Image
from ddpm2 import UNet, ddim_sample, make_cond, DEVICE
from metrics import compute_metrics

CKPT = "checkpoints/ddpm2_ep25.pt"
# vf calibration (epoch 25, guidance 3): request -> achieved. Reliable ~0.37-0.57.
CAL_REQ = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
CAL_ACH = np.array([0.367, 0.400, 0.442, 0.488, 0.529, 0.567])


def request_for(target_vf):
    """Invert the measured achieved->request map so the model hits the target."""
    t = float(np.clip(target_vf, CAL_ACH[0], CAL_ACH[-1]))
    return float(np.interp(t, CAL_ACH, CAL_REQ))


def load_model(path=CKPT):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = UNet().to(DEVICE)
    m.load_state_dict(ck["ema"])               # EMA weights
    return m


def generate(target_vf, stiff="high", n=16, tol=0.05, guidance=3.0, steps=40,
             symmetrize=False, max_batches=12, batch=16, model=None, seed=0):
    model = model or load_model()
    req = request_for(target_vf)
    rng = np.random.default_rng(seed)
    kept, tried = [], 0
    for _ in range(max_batches):
        cells = ddim_sample(model, batch, make_cond(req, stiff), steps=steps,
                            guidance=guidance, symmetrize=symmetrize,
                            seed=int(rng.integers(0, 10**6)))
        for c in cells:
            tried += 1
            m = compute_metrics(c, do_fem=False)
            if (abs(m["volume_fraction"] - target_vf) <= tol
                    and m["periodic_connected_x"] and m["periodic_connected_y"]):
                kept.append(c)
        if len(kept) >= n:
            break
    return kept[:n], tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", type=float, required=True)
    ap.add_argument("--stiff", default="high", choices=["low", "medium", "high"])
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--symmetrize", action="store_true")
    ap.add_argument("--out", default="generated_v2")
    args = ap.parse_args()
    cells, tried = generate(args.vf, args.stiff, args.n, args.tol,
                            args.guidance, symmetrize=args.symmetrize)
    os.makedirs(args.out, exist_ok=True)
    for i, c in enumerate(cells):
        Image.fromarray((c * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"gen_vf{args.vf}_{args.stiff}_{i:03d}.png"))
    yld = 100 * len(cells) / max(tried, 1)
    print(f"accepted {len(cells)}/{tried} ({yld:.0f}% yield) -> {args.out}")
    if len(cells) < args.n:
        print(f"got {len(cells)} of {args.n}; raise --tol or --max-batches, or "
              f"pick a vf nearer 0.37-0.57 (the model's reliable range)")


if __name__ == "__main__":
    main()
