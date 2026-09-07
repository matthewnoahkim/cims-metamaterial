"""Head-to-head: diffusion warm start vs. a fair cold-start full TO.

Overlays the diffusion warm-start correction-budget curve (from study_results_v2)
against the robust-projection cold-start full TO (full_to_baseline) on one
FE-solve axis, and reports the cost to reach a target stiffness for each.
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import full_to_baseline as ftb

HERE = os.path.dirname(os.path.abspath(__file__))

# --- diffusion warm start (std path, continuous init) from the v2 study
R = list(csv.DictReader(open(os.path.join(HERE, "study_results_v2.csv"))))
for r in R:
    r["k"] = int(r["k"]); r["E"] = float(r["E"])
def med(arm, k):
    v = [r["E"] for r in R if r["arm"] == arm and r["k"] == k]
    return float(np.median(v)) if v else np.nan
ITERS = max(r["k"] for r in R if r["arm"] == "warm_cont")
warm_fe = [0] + [3 * k for k in range(1, ITERS + 1)]
warm_E = [med("diffusion_raw", 0)] + [med("warm_cont", k) for k in range(1, ITERS + 1)]

# --- fair cold-start full TO at matched (32) and higher (48) resolution
print("running cold-start full TO (this takes ~1-2 min)...", flush=True)
_, i32 = ftb.cold_to(32, 0.5, iters=150, seed=0)
_, i48 = ftb.cold_to(48, 0.5, iters=150, seed=0)
cold32_fe = [0] + [t[0] for t in i32["traj"]]
cold32_E = [0.0] + [t[1] for t in i32["traj"]]

# --- report
warm_conv = warm_E[-1]
def budget_to(curve_fe, curve_E, target):
    return next((fe for fe, e in zip(curve_fe, curve_E) if e >= target), None)
tgt = 0.95 * warm_conv
print(f"\ndiffusion warm start (32): converged E {warm_conv:.3f}; 95% ({tgt:.3f}) by {budget_to(warm_fe, warm_E, tgt)} FE solves")
print(f"cold full TO (32): E {i32['E']:.3f} at {i32['fe_solves']} FE solves; reaches {tgt:.3f} by {budget_to(cold32_fe, cold32_E, tgt)} FE solves")
print(f"cold full TO (48): E {i48['E']:.3f} at {i48['fe_solves']} FE solves (higher res the 32x32 model can't reach)")
if budget_to(cold32_fe, cold32_E, tgt) and budget_to(warm_fe, warm_E, tgt):
    print(f"=> speed advantage: {budget_to(cold32_fe, cold32_E, tgt)/budget_to(warm_fe, warm_E, tgt):.0f}x fewer FE solves to the same stiffness")

# --- plot
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(warm_fe, warm_E, "o-", color="#058DC7", lw=2, label="diffusion warm start + SIMP (32×32)")
ax.plot(cold32_fe, cold32_E, "s--", color="#B03A2E", lw=2, label="cold full TO, robust projection (32×32)")
ax.axhline(i48["E"], color="#7D3C98", ls=":", lw=1.5, label=f"cold full TO (48×48) = {i48['E']:.3f}")
ax.axhline(med("diffusion_raw", 0), color="#999", ls=":", lw=1.2, label="diffusion raw, no correction")
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("physics-evaluation budget  (FE solves, symlog)")
ax.set_ylabel("effective stiffness  E   (median)")
ax.set_title("Light correction vs. full TO from scratch")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "compare_baseline.png"), dpi=150)
print("\nwrote compare_baseline.png")
