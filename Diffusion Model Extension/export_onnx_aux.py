"""
ONNX export of the auxetic specialist DDPM (ddpm_aux, cond = [vf, nu]).

Graph interface (same pattern as ddpm2's export, narrower cond):
  Inputs:  x [b,1,32,32] f32, t [b] int64, cond [b,2] f32
  Outputs: eps_c [b,1,32,32], eps_u [b,1,32,32]
Browser does CFG: eps = eps_u + w*(eps_c - eps_u).

Verifies per-step parity vs PyTorch and the full DDIM loop (numpy scheduler +
onnxruntime vs the PyTorch sampler, mean IoU).

Usage:  python export_onnx_aux.py --ckpt checkpoints/ddpmaux.pt --out model/ddpm_aux_ep80.onnx
"""
import os, argparse
import numpy as np
import torch, torch.nn as nn
from ddpm_aux import UNet, COND_DIM, IMG, T_STEPS, abar, make_cond, ddim_sample, DEVICE


class Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, t, cond):
        return self.m(x, t, cond), self.m.forward_uncond(x, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="model/ddpm_aux_ep80.onnx")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = UNet(); model.load_state_dict(ck["ema"]); model.eval()
    wrap = Wrapper(model).eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    b = 2
    dummy = (torch.randn(b, 1, IMG, IMG),
             torch.randint(0, T_STEPS, (b,), dtype=torch.int64),
             torch.randn(b, COND_DIM))
    torch.onnx.export(
        wrap, dummy, args.out, opset_version=17,
        input_names=["x", "t", "cond"], output_names=["eps_c", "eps_u"],
        dynamic_axes={k: {0: "b"} for k in ["x", "t", "cond", "eps_c", "eps_u"]},
        dynamo=False)
    print(f"exported {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    torch.manual_seed(1)
    x = torch.randn(4, 1, IMG, IMG)
    t = torch.tensor([399, 250, 120, 5], dtype=torch.int64)
    c = torch.tensor([make_cond(0.47, -0.6)] * 4, dtype=torch.float32)
    with torch.no_grad():
        ec_t, eu_t = wrap(x, t, c)
    ec_o, eu_o = sess.run(None, {"x": x.numpy(), "t": t.numpy(), "cond": c.numpy()})
    print(f"per-step parity: cond {np.abs(ec_t.numpy()-ec_o).max():.2e}  "
          f"uncond {np.abs(eu_t.numpy()-eu_o).max():.2e}")

    n, steps, w = 16, args.steps, args.guidance
    cond = make_cond(0.47, -0.6)
    x0_pt = ddim_sample(model.to(DEVICE), n, cond, steps=steps, guidance=w, seed=0)

    ab = abar.numpy()
    ts = np.linspace(T_STEPS - 1, 0, steps).astype(np.int64)
    torch.manual_seed(0)
    x = torch.randn(n, 1, IMG, IMG).numpy()
    cnp = np.tile(np.array(cond, np.float32), (n, 1))
    for i in range(steps):
        t_arr = np.full(n, ts[i], np.int64)
        ec, eu = sess.run(None, {"x": x.astype(np.float32), "t": t_arr, "cond": cnp})
        eps = eu + w * (ec - eu)
        a_t = ab[ts[i]]
        x0 = np.clip((x - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t), -1, 1)
        x = (np.sqrt(ab[ts[i+1]]) * x0 + np.sqrt(1 - ab[ts[i+1]]) * eps) \
            if i < steps - 1 else x0
    x0_ox = ((x[:, 0] + 1) / 2 >= 0.5).astype(np.uint8)

    ious = []
    for a, bb in zip(x0_pt, x0_ox):
        inter = np.logical_and(a, bb).sum(); union = np.logical_or(a, bb).sum()
        ious.append(inter / union if union else 1.0)
    print(f"full-loop parity: mean IoU {np.mean(ious):.4f} (min {np.min(ious):.4f})")


if __name__ == "__main__":
    main()
