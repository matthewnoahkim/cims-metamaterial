# Generative AI for Topology-Optimized Metamaterial Unit Cells — Technical Write-Up

*Matthew Kim — CIMS Lab (Computing, Intelligence & Mechanics), University of Minnesota; PI: Dr. Qizhi He*

## Project summary

- Built an end-to-end pipeline that generates 2D mechanical metamaterial unit cells with **targeted properties** (volume fraction, stiffness class, and — after the latest extension — negative Poisson's ratio), using conditional denoising diffusion models, and validates every generated design with real physics rather than pixel similarity.
- The entire product runs **in the browser**: the trained diffusion models execute via ONNX Runtime Web (WebGPU/WASM), and each generated cell is scored live by a JavaScript port of a periodic-homogenization finite-element solver.
- Core principle throughout: **measure, never assume.** Labels come from FEM, generated designs are re-scored by the same FEM, model checkpoints are selected by measured controllability, and every claim in the UI states its honest limits.

## Research question

- Can a generative model learn from topology-like designs and produce *new* unit cells that hit a requested mechanical target — and can that control be extended to auxetic behavior (negative Poisson's ratio), which requires generating a categorically different geometry family?
- A metamaterial's behavior comes from geometry, not chemistry: the same base solid, arranged differently, can be stiff, soft, or auxetic. A unit cell is stored as a binary 32×32–64×64 image (solid/void); tiling it builds the material, so one small image determines the effective elastic tensor.

## Stage 1 — Dataset (8,040 cells)

- Procedurally generated six geometry families (holes, lattice, cross, ring, bars, reentrant) as binary 64×64 images — a synthetic stand-in for true topology-optimization output, and documented as such.
- Labels are **measured from the image**, not taken from the generator: volume fraction, connected components, spanning/periodic percolation (via 3×3 tiling test), and mirror-symmetry class.
- Early-stage honesty flag that paid off later: the "auxetic_like" label was set *by construction* from the reentrant family — and Stage "auxetic extension" below proved that label physically wrong.

## Stage 2 — Physics evaluator (periodic homogenization)

- Implemented a plane-stress Q4 finite-element homogenizer with periodic boundary conditions: solves three unit macro-strain load cases, assembles the effective stiffness tensor CH, and derives Ex, Ey, ν, G, K, plus manufacturability metrics (thin members, trapped voids, floating material).
- **Validation-first discipline** — the evaluator earned trust before judging any model output:
  - solid cell → E = 1.000, ν = 0.300 exactly;
  - layered cell → falls between the Voigt/Reuss analytical bounds;
  - solid cell of auxetic *base material* (ν₀ = −0.3) → recovers −0.3 exactly.
- The same evaluator labels training data and scores generated cells, so "accuracy" means performance error against target, not reconstruction error.

## Stage 3 — Baseline: conditional VAE

- Reconstructs cells at IoU 0.98 but fragments at low density, and its apparent stiffness advantage was traced to a bias: it overshoots density (vf error 0.138), so higher E_eff is an artifact of extra material, not better structure. This diagnosis shaped how later comparisons were framed.

## Stage 4 — Conditional DDPM (main model, "v2")

- Small U-Net (~2.0M parameters) at 32×32; DDPM with 400-step linear β schedule; DDIM sampling (50 steps); classifier-free guidance (cond dropout p = 0.12, guidance w = 3); EMA weights.
- Conditioning: `[volume_fraction, stiffness-class one-hot(3)]` through an embedding MLP added to the timestep embedding.
- **Key methodological finding:** checkpoint selection by *controllability probe*, not loss — epoch 29 had lower loss than epoch 25 but measurably weaker conditioning. Shipped epoch 25.
- Measured behavior: vf response monotonic but compressed (requests 0.2–0.7 → achieved 0.37–0.57), handled with a measured request→achieved **calibration map** inverted at run time; stiffness conditioning separates low vs high E by ~2×; 22–24 of 24 samples periodically connected.

## Stage 5 — Validation and comparison

- Internal comparison at vf = 0.50 (all scored by the same evaluator): diffusion is the balanced choice — vf error 0.013 (vs 0.138 VAE), 100% connectivity (vs 44% random), E_eff 0.095; manual designs remain stiffest (0.274).
- External comparison kept strictly at the **aggregate-metric level**: this work's 1.3% vf error sits between Kollmann's CNN (1.2%) and TopoDiff (1.85%), on different datasets/tasks/hardware — deliberately *not* claimed as a head-to-head win, since no matched target pairs exist. DiffMat was excluded from the accuracy chart because its error metric (distance to a stress–strain curve) is a different quantity; plotting it would be a false equivalence.

## Stage 6 — Browser live demo

- Two static pages, no build step, deployable on GitHub Pages:
  - **Diffusion page**: ONNX Runtime Web runs the exported denoiser (8 MB, opset 17); the DDIM loop, classifier-free guidance, thresholding, and symmetry folding are hand-written JavaScript; the JS schedule was verified to match the Python schedule exactly.
  - **Procedural page**: instant candidates from a stochastic generator (Gaussian random fields + strut motifs) scored by the same in-browser evaluator — a zero-download fallback.
- Export verification: per-step ONNX-vs-PyTorch parity ~2e-6; full 50-step sampling loop (numpy scheduler + ONNX) matches the PyTorch sampler at 0.9989 IoU.
- The in-browser homogenizer (matrix-free preconditioned conjugate-gradient, softened void) was validated against the Python solver on the same analytical cases.

## Auxetic extension, part 1 — physics discovery and browser controls

- **Discovery that reframed the problem:** the dataset's "reentrant" bowtie family measures ν ≈ **+0.23** under the project's own homogenizer — auxetic-like by label only. The evaluator's celebrated "−0.3 recovered exactly" validation was a solid cell of auxetic *base material*, not a geometric auxetic. No component of the pipeline had ever actually produced ν < 0.
- Prototyped candidate auxetic topologies in Node.js against the exact browser evaluation path: pixelated re-entrant honeycombs stay ν > 0 at this resolution (walls too thick relative to the cell), but **rotating squares with corner hinges** — four rigid squares, alternately rotated ±α, tied by small hinge disks at lattice-edge midpoints — reliably homogenize to ν as low as **−0.65** at vf 0.40–0.55.
- **Resolution-artifact hunt:** drawing candidates at 48×48 and downsampling to the 24×24 evaluation grid made ν flip sign between resolutions — hinge geometry is a ~1-pixel feature. Fix: draw the family **natively on the evaluation grid** and upsample ×2 for display, so the displayed cell is exactly what the FEM sees; results then agree between 24×24 and a 48×48 refined solve (e.g., −0.66 vs −0.61).
- Shipped controls:
  - *Procedural page*: "Auxetic (target ν)" goal with a target slider (−0.7…+0.3); square size is bisected to hit the requested vf exactly; live browser test at target −0.30 gave 17/40 candidates measured auxetic, best cell ν = −0.189 and connected.
  - *Diffusion page (at that point)*: an honest "auxetic search" mode — the v2 model has no ν channel, so it double-sampled and ranked by measured ν, reporting that the minimum found is usually still positive (measured 0/16 auxetic) and pointing to the procedural page.

## Auxetic extension, part 2 — retraining the DDPM

- **Training data**: generated 2,000 rotating-squares cells drawn native at 32×32 (the DDPM's resolution) and saved at 64×64 so the training resize is lossless; every cell FEM-labeled by the Stage-2/3 evaluator — 63% measured auxetic, ν down to ≈ −1.0 (the ideal-mechanism limit). Deduplicated by image hash; parameters randomized (rotation angle, per-square jitter, hinge radius, vf via bisection).
- **Four systematically-diagnosed failed attempts** to add ν control to the mixed 10,040-cell model (each ~20-minute run on an RTX 5060, probed with FEM):
  1. *Warm start + scalar ν channel* (zero-initialized column, verified bit-exact against v2 at start): ν channel stayed inert through 30 epochs, and the inherited vf conditioning *decayed* as fine-tuning progressed.
  2. *Same, from scratch, with an explicit auxetic flag and 5× oversampling*: still inert; generated ν ≈ dataset mean regardless of request.
  3. *TopoDiff-style spatial conditioning planes* (constant [vf, ν, flag, keep-mask] channels concatenated to the input): sample quality and connectivity collapsed within this training budget.
  4. *Auxetic-class one-hot* (extending the mechanism that demonstrably works for stiffness classes): aux classes produced ν ≈ +0.04, ≤25% auxetic hits.
- **Control experiment that isolated the cause:** retraining an exact v2 replica in the same environment reproduced the shipped model's vf response almost exactly (0.366/0.444/0.533 vs 0.358/0.444/0.523), ruling out code/environment regressions. Conclusion: the small model's global conditioning can bias *statistics* (vf, ~2× stiffness) but cannot switch geometry *family* on demand.
- **Shipped architecture — a two-expert system:** a dedicated auxetic specialist DDPM (same U-Net backbone, cond = `[vf, ν]`) trained only on the rotating-squares family, where ν→geometry is a smooth, learnable mapping; the proven v2 model keeps serving standard requests; the demo routes by the sign of the requested ν.
- Supporting findings along the way:
  - Random periodic-roll augmentation (physically a no-op) *hurt* conditioning; aligned cells train better.
  - **Conditioning strength peaks mid-training and decays** — the specialist peaked at epoch 120 of 240 (67% auxetic at request −0.8, best-of-batch to −1.7, vs 12% by epoch 240). Selected epoch 120 via controllability probe — the v2 lesson, independently reproduced.
  - The 32→24 evaluation downsample in the browser was replaced by native 32×32 homogenization, since it distorted exactly the hinge pixels that determine ν.
- **Verification and measured results:**
  - ONNX export bit-consistent with PyTorch: per-step parity ~5e-6; full 50-step DDIM loop mean IoU **1.0000** with matched RNG (an initial 0.77 IoU was traced to comparing CUDA-seeded vs CPU-seeded noise — a verification bug, not an export bug).
  - In-browser end-to-end test at target ν = −0.50: 16/16 periodically connected, 8/16 measured auxetic, best cell **ν = −0.398**, ranked by homogenized ν against the target.
  - Measured single-design inference: **0.38 s** (RTX 5060 laptop GPU, 50 DDIM steps) / 0.52 s CPU — replacing the poster's unmeasured "~4 s" estimate with a timed number.
  - Honest limits stated in the UI: the specialist's vf control is approximate (generated vf ≈ 0.55 ± 0.04), and the ν slider steers the sampling distribution while every displayed number is homogenized from the actual cell.

## Engineering lessons (the process, distilled)

- **Validate the validator first.** Trust in every downstream number rests on the evaluator reproducing analytical cases; the same principle exposed the "auxetic-like" label as physics-free.
- **Labels by construction are hypotheses; labels by measurement are data.** The reentrant family's ν ≈ +0.23 was found only because generated cells are always re-measured.
- **Select checkpoints by the property you care about.** Loss kept improving while conditioning degraded — observed independently in three separate training campaigns.
- **Chase resolution artifacts before believing results.** ν flipped sign under downsampling twice (procedural drawing grid, browser evaluation grid); both fixed by evaluating the exact displayed geometry.
- **When a capability won't train in, isolate it.** Four falsified conditioning designs plus one clean control experiment justified the two-expert architecture with evidence rather than intuition.
- **Verify exports numerically, and verify the verification.** Per-step parity, full-loop IoU with matched RNG streams, and a browser smoke test on both model paths.
- **State limits where users see them.** Compressed vf range, approximate vf in auxetic mode, and filter-vs-conditioning distinctions are all in the UI copy, not buried in notes.

## Stack

- **Modeling/training**: PyTorch 2.11 (CUDA 12.8, RTX 5060), DDPM/DDIM, classifier-free guidance, EMA; ~30 s/epoch on the 10k-cell set.
- **Physics**: NumPy/SciPy sparse FEM homogenization (Python); matrix-free Jacobi-preconditioned CG port in vanilla JavaScript.
- **Deployment**: ONNX (opset 17) via torch.onnx; ONNX Runtime Web with WebGPU→WASM fallback; static HTML/JS, no framework, no build step.
- **Verification tooling**: Node.js harnesses replicating the browser evaluator bit-for-bit for offline physics sweeps.

## Current limitations and next steps

- Training data remains synthetic; next step is real SIMP/homogenization-based topology-optimization data (e.g., Xia & Breitkopf's energy-based code, the same family used by Kollmann and DiffMat).
- Guidance is still generate-then-filter at the batch level; moving regressor/classifier guidance inside the sampler (TopoDiff-style) is the stated next step.
- The specialist's vf channel is weak; candidate fixes include longer/staged training, vf-stratified sampling, or conditioning on full stiffness components (E1111, E1122, E1212) to target anisotropy and ν jointly.
- Outputs remain speckly at family boundaries; density-contour extraction + Ramer–Douglas–Peucker smoothing (as in DiffMat) is the planned cleanup for manufacturability.
- An example-level head-to-head with a published design (matched target, side-by-side) would upgrade the aggregate-only external comparison.
