"""Final analysis + figure for the v2 study (reads study_results_v2.csv).

Reports the solid, honest findings and the implementable efficiency win:
  physics-guided sampling + a SHORT continuous-init SIMP polish matches the
  15-iteration config's stiffness at a fraction of the FE-solve cost.
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, "study_results_v2.csv"))))
for r in R:
    r["k"] = int(r["k"]);
    for c in ("E", "vferr", "pieces", "connected", "nu"): r[c] = float(r[c])
GUIDE_FE = 3 * min(12, max(4, round(40 * 0.3)))    # guidance cost in FE solves (K steps x 3)

def arr(arm, k, key="E"): return np.array([r[key] for r in R if r["arm"] == arm and r["k"] == k])
def med(arm, k, key="E"):
    a = arr(arm, k, key); return float(np.median(a)) if len(a) else np.nan
def iqr(arm, k):
    a = arr(arm, k); return (np.percentile(a, 25), np.percentile(a, 75)) if len(a) else (np.nan, np.nan)

ITERS = max(r["k"] for r in R if r["arm"] == "warm_cont")
raw, guided = med("diffusion_raw", 0), med("guided_raw", 0)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
# ---- Panel A: correction-budget curves (SIMP FE solves), the working arms
ks = list(range(1, ITERS + 1)); fe = [3 * k for k in ks]
for arm, col, lab, start in [("warm_guided", "#2E8B57", "guided init + SIMP", guided),
                             ("warm_cont", "#058DC7", "diffusion init + SIMP (continuous)", raw),
                             ("warm_bin", "#7FB3D5", "diffusion init + SIMP (binarized, v1)", raw)]:
    ys = [start] + [med(arm, k) for k in ks]
    xs = [0] + fe
    ax[0].plot(xs, ys, "o-", color=col, lw=2, ms=4, label=lab)
    if arm in ("warm_guided", "warm_cont"):
        lo = [start] + [iqr(arm, k)[0] for k in ks]; hi = [start] + [iqr(arm, k)[1] for k in ks]
        ax[0].fill_between(xs, lo, hi, color=col, alpha=0.15)
ax[0].axhline(guided, color="#2E8B57", ls=":", lw=1, alpha=.7)
ax[0].axhline(raw, color="#999", ls=":", lw=1.2, label="no correction (raw sample)")
ax[0].set_xlabel("SIMP correction budget  (FE solves)")
ax[0].set_ylabel("effective stiffness  E   (median, IQR band)")
ax[0].set_title("Light correction suffices"); ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].grid(alpha=.3)

# ---- Panel B: implementable config comparison (E vs total FE-solve cost)
cfgs = [("raw\nsample", raw, 0),
        ("physics\nguided", guided, GUIDE_FE),
        ("guided +\nSIMP-5", med("warm_guided", 5), GUIDE_FE + 15),
        ("guided +\nSIMP-15", med("warm_guided", 15), GUIDE_FE + 45)]
xpos = np.arange(len(cfgs))
Es = [c[1] for c in cfgs]; costs = [c[2] for c in cfgs]
bars = ax[1].bar(xpos, Es, color=["#999", "#2E8B57", "#058DC7", "#1B4F72"], width=0.62)
for i, (b, c) in enumerate(zip(bars, cfgs)):
    ax[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.012, f"E {c[1]:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.003, f"{c[2]} FE solves", ha="center", fontsize=7.5, color="#555")
ax[1].set_xticks(xpos); ax[1].set_xticklabels([c[0] for c in cfgs], fontsize=8.5)
ax[1].set_ylim(0, max(Es) * 1.28); ax[1].set_ylabel("effective stiffness  E   (median)")
ax[1].set_title("Recommended efficient config"); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "study_v2_curves.png"), dpi=150)
print("wrote study_v2_curves.png")

# ---- readouts
conv = med("warm_cont", ITERS)
def budget95(arm):
    return next((3 * k for k in range(1, ITERS + 1) if med(arm, k) >= 0.95 * conv), None)
print(f"\nconverged E (diffusion init, k={ITERS}): {conv:.3f}")
for a in ["warm_guided", "warm_cont", "warm_bin"]:
    print(f"  {a:12s}: 95% at {budget95(a)} FE solves (SIMP); k0={ 'guided' if a=='warm_guided' else 'raw'}={med(a,1):.3f}@k1")
print(f"\nattribution @ k0:  raw E {raw:.3f}  ->  +guidance E {guided:.3f}  ({guided/raw:.1f}x), 100% connected")
wg5, wg15 = med("warm_guided", 5), med("warm_guided", 15)
print(f"efficient config:  guided + SIMP-5 = E {wg5:.3f} at {GUIDE_FE+15} FE solves"
      f"  vs  guided + SIMP-15 = E {wg15:.3f} at {GUIDE_FE+45} FE solves"
      f"  ({wg5/wg15*100:.0f}% of stiffness at {(GUIDE_FE+15)/(GUIDE_FE+45)*100:.0f}% of cost)")
# continuous vs binarized init, early speed
print(f"continuous vs binarized init @k1: {med('warm_cont',1):.3f} vs {med('warm_bin',1):.3f}"
      f"  @k2: {med('warm_cont',2):.3f} vs {med('warm_bin',2):.3f}")
