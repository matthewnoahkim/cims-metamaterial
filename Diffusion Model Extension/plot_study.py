"""Plot the attribution / correction-budget study from study_results.csv."""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, "study_results.csv"))))
for r in R:
    for k in ("k", "fe_solves", "pieces", "connected"):
        r[k] = float(r[k])
    for k in ("E", "bulk", "nu", "vf", "vferr"):
        r[k] = float(r[k])

def sel(arm, k):
    return [r for r in R if r["arm"] == arm and int(r["k"]) == k]
def med(arm, k, key="E"):
    s = sel(arm, k)
    return float(np.median([r[key] for r in s])) if s else np.nan
def mean(arm, k, key="E"):
    s = sel(arm, k)
    return float(np.mean([r[key] for r in s])) if s else np.nan
def rate(arm, k):
    s = sel(arm, k)
    return float(np.mean([r["connected"] for r in s])) if s else np.nan

ITERS = int(max(r["k"] for r in R))
ks = list(range(0, ITERS + 1))
fe = [3 * k for k in ks]
d0 = med("diffusion_despeckle", 0)                 # warm start begins here (k=0)
warmE = [d0] + [med("warm_simp", k) for k in range(1, ITERS + 1)]
coldE = [0.0] + [med("cold_simp", k) for k in range(1, ITERS + 1)]
warmC = [rate("diffusion_despeckle", 0)] + [rate("warm_simp", k) for k in range(1, ITERS + 1)]
coldC = [0.0] + [rate("cold_simp", k) for k in range(1, ITERS + 1)]
rawE = med("diffusion_raw", 0); guidedE = med("guided_raw", 0)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(fe, warmE, "o-", color="#058DC7", lw=2, label="warm start — diffusion sample")
ax[0].plot(fe, coldE, "s--", color="#B03A2E", lw=2, label="cold start — uniform (full TO)")
ax[0].axhline(guidedE, color="#2E8B57", ls=":", lw=1.5, label="physics-guided, no SIMP")
ax[0].axhline(rawE, color="#999", ls=":", lw=1.5, label="diffusion raw, no correction")
ax[0].set_xlabel("correction budget  (FE solves)"); ax[0].set_ylabel("effective stiffness  E  (median)")
ax[0].set_title("Stiffness vs. correction budget"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(fe, [100*v for v in warmC], "o-", color="#058DC7", lw=2, label="warm start — diffusion")
ax[1].plot(fe, [100*v for v in coldC], "s--", color="#B03A2E", lw=2, label="cold start — uniform")
ax[1].set_xlabel("correction budget  (FE solves)"); ax[1].set_ylabel("periodically connected  (% of samples)")
ax[1].set_title("Connectivity vs. correction budget"); ax[1].set_ylim(-5, 105)
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "study_curves.png"), dpi=150)
print("wrote study_curves.png")

# reliability stats at the full budget
def arr(arm, k, key="E"):
    return np.array([r[key] for r in sel(arm, k)])
wf, cf = arr("warm_simp", ITERS), arr("cold_simp", ITERS)
wins = int(np.sum(wf > cf))
print(f"\nAt k={ITERS} ({3*ITERS} FE solves):")
print(f"  warm start E: {wf.mean():.3f} +/- {wf.std():.3f}")
print(f"  cold start E: {cf.mean():.3f} +/- {cf.std():.3f}")
print(f"  warm > cold in {wins}/{len(wf)} paired samples")
# budget to reach 90% of warm's converged E
tgt = 0.9 * warmE[-1]
kreach = next((3*k for k in range(1, ITERS+1) if med("warm_simp", k) >= tgt), None)
print(f"  warm reaches 90% of converged E ({tgt:.3f}) by {kreach} FE solves")
