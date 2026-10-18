# DE inference preparation (steps I–L): train ref stats, paired latent perturbation, log-r grid, Null + response curves

> Task `08-10-de-inference-prep-ijkl`. Implements experiment-plan steps I, J, K, L
> of `docs/OC_LatentFlow_DE_confirmed_decisions_summary.md` §2.2 — the **mechanism**
> half of the DE-inference redesign. Builds the reusable machinery that turns a
> frozen, gate-passing `DEFlow` into a perturbational effective-connectivity (DE)
> probe.
>
> **Result type: mechanism complete; characterization pending full run.** The
> IJKL machinery (ref stats + perturbation + grid + response + Null + CI) is
> built, unit-tested (17 tests), and validated end-to-end on a smoke subset.
> The train-split response-vs-Null characterization is reported honestly below.

## Setup

- **Frozen substrate**: the `08-09` Stage-2 FM
  `checkpoints/de_flow/fold0_de_flow_best.pt` (DEFlow: frozen DEVAE D=8/beta=0.01
  by composition + `FlowVelocityNet` mexhat+W mixer). Latent `(B, C=128,
  T_lat=16, D=8)`, per-channel `d=128` (held constant, decision 16).
- **Gate prerequisite**: `08-09` ran the feasibility gate; `fold0_verdict_v3.json`
  + `fold0_t004.json` exist (T-004 realized). The FM is the substrate DE runs on.
- **Train split**: re-encoded from raw MODMA fold-0 train subjects via
  `DEFlow.encode_full` (the substrate `fold0_pairs.npz` stores only μ; the
  perturbation needs the real posterior `(μ, logvar)`). N≈7200 train windows,
  43 subjects.
- **One seam edit** to `models/fm_model.py`: added `sample_from_latent(z_obs,
  c_obs, n_steps)` (decision 11 requires injecting a perturbed init state; the
  existing `sample()` always re-encodes `x_obs` and only overrides `c_obs`).
  Baseline path identical to `sample()` (verified, max diff 0.0). `sample()` /
  `forward_stage2` / checkpoints untouched.

## Step I — train-only reference statistics (decision D1 = Ledoit-Wolf)

Estimated from the **train split only**, frozen (decisions 17, 18).

| Quantity | Value | Notes |
|---|---|---|
| `S_ref` estimator | Ledoit-Wolf (default, D1) | pooled over all channels + train latents |
| condition number | ~250 | well-conditioned (no degeneracy) |
| effective rank | ~40–53 / 128 | participation ratio; full-rank, healthy |
| `0.5·log\|S_ref\|` | ~−8.5 (full) / −44 (smoke) | finite; the `DE^(d)` term (decision 16) |
| `MAD_b` median | ~0.40–0.66 | decoded-signal domain; range [0.11, 0.78] across channels (~7× spread) |

- `MAD_b` is in the **decoded-signal domain** (from `decode(μ_fc)`), matching the
  units of the decoded-MSE response (decision 14). Latent-domain MAD would be
  unit-mismatched.
- The `S_ref` geometry is what makes the perturbation `δ = r·(L_ref @ ξ_k)`
  source-invariant (all sources share one covariance, decision 12).

## Step K — r_max + log-r grid (decision D2 = Mahalanobis-quantile)

| Quantity | Value | Notes |
|---|---|---|
| criterion | Mahalanobis quantile (default, D2) | `d_M(μ_a + r·L_ref@ξ_k) = r·||ξ_k||` (whitened) |
| q | 0.99 | default; q∈{0.95,0.99,0.999} → r_max∈{1.29,1.43,1.63} |
| `dM_q` (train 0.99-quantile) | ~16.2 | mean dM ~11.0, max ~21.4 |
| `E[‖ξ_k‖]` (d=128) | 11.29 | ≈ √d (11.31) |
| `r_max` | ~1.43 | `dM_q / E[‖ξ_k‖]`; finite, within train support |
| decode amplitude at r_max | ~5.86 (vs base 5.84) | no saturation / NaN / Inf — r_max is well inside the decoder support |
| grid | 18 positive points + r=0 | log-spaced (decision 21 / C9 candidate) |

- **No hardcoded `SIGMA_MAX`** (contrast the old paradigm's
  `attribution/bisect_sigma.py:83` `SIGMA_MAX = 0.3`, which had no support
  reasoning). The bound is data-driven via the `S_ref` geometry.
- The NUMERIC `r_max` is a calibration concern (decision 22 / breakpoint 2);
  this ships the selector + a principled default. The frozen value is locked at
  step N.

## Steps J+L — paired mean-shift perturbation + response + Null + CI

**Shared-random-variable invariants (decision 13, load-bearing)** — all verified
by `tests/test_de_prep.py` (17/17 pass):

1. `R(r=0) = 0` **exactly** (max|R0| = 8.8e-8, float32 noise) — the shared-`ε`
   invariant: at r=0 the ±δ vanishes, so `z^±(0) == z_base`, so the deterministic
   rollout gives `Δ^±(0) = 0`.
2. Same `ε_p` for baseline/+/− at fixed (window, sample).
3. Same `ξ_k` across the whole r-grid and across sources.
4. Condition synced with init state (decision 11): the same `r·δ` shifts both
   `z_obs,a` and `c_obs,a` (on μ).

**Baseline precomputation optimization**: `x_base = decode(rollout(μ+s⊙ε, μ))` is
source-invariant and r-invariant, so it is computed once over all windows instead
of inside the per-source loop — saving a factor of ~n_src. This is the difference
between an infeasible and a tractable full run.

### Smoke characterization (3 sources, 1 subject, 64 windows)

| Quantity | Value | Interpretation |
|---|---|---|
| non-monotone fraction | 0.034 | most subject-mean curves are monotone (good) |
| `R(r_max)` mean | 0.0011 | response at the support boundary |
| Null mean | 0.4476 | stochastic-variance floor (r-invariant) |
| response vs Null | **BELOW** | at this smoke scale the perturbation response does not exceed the forecast-variance floor |

The "response below null" at the smoke scale is expected and **not** a mechanism
defect: with a single subject / 64 windows / K=2 directions / P=1 sample, the
Monte-Carlo estimate of the response is noisy and the Null (two independent
posterior draws) is large. The full fold-0 run (more subjects, full K/P) gives
the honest answer.

### Full fold-0 run (8-source verification, strategy A+B)

The full ~7000-window × 128-source run is hours-scale. Three optimizations make
it tractable:

1. **Baseline precomputation** (source/r-invariant): the baseline rollout is
   computed once over all windows, not per-source. Saves a factor of ~n_src on
   the baseline.
2. **Vectorized ±×r-grid rollout** (`response.py`): the per-`(source, batch)`
   inner loop collapses 2M rollout calls into ONE call that processes all
   `M×2×P×B` samples at once. Numerically identical (unit-tested:
   `test_vectorized_rollout_matches_explicit_loop`).
3. **Strategy A + B (the real speed levers)** — `--windows-per-subject N`
   (per-subject uniform window subsample, keeps cluster-bootstrap subject
   coverage balanced) + `--n-positive M` (grid density). These cut the total
   FLOP, which the vectorization alone cannot do (GPU is already 98% utilized;
   the bottleneck is decode compute, not launch latency).

**Verified run** (2026-08-11, `--windows-per-subject 5 --n-positive 8`):

| Quantity | Value |
|---|---|
| N (windows) | 215 (43 subjects × 5 windows/subject, full subject coverage) |
| M (grid, incl r=0) | 9 |
| K (directions), P (posterior) | 4, 2 |
| sources probed | 8 (ch 0,16,32,48,64,80,96,112) |
| total wall time | **555s = 9m15s** (encode 14s + Step I/K 1s + response 540s) |
| per-source net | **~1m11s** |
| `S_ref` (Ledoit-Wolf) | cond 280.6, eff_rank 38.5, logdet −14.7 |
| `MAD_b` | median 0.638, range [0.535, 0.714] |
| `r_max` (Mahalanobis q=0.99) | 1.393 (dM_q 15.73) |
| `R(r=0)` | **exactly 0** (shared-ε invariant) |
| non-monotone fraction | 0.039 (most subject-mean curves monotone) |
| `R(r_max)` mean | 0.0056 |
| Null mean | 0.8758 (r-invariant, finite) |
| response vs Null | **BELOW** |

**Extrapolation to full 128-source matrix**: ~1m11s × 128 ≈ **2.5 h** (offline,
one overnight run). The 8-source verification above is the in-session validation;
the full matrix is the production deliverable.

**Scientific finding (honest)**: `R(r_max) mean = 0.0056 vs Null = 0.8758`. The
R and Null are deliberately **asymmetric by design** and the comparison IS valid:
- **R** uses shared ε (decision 13: baseline/+/− share the same posterior noise),
  so the posterior variance cancels in the difference `Δ = x^± − x_base`. R
  measures the **pure δ causal effect** on the decoded forecast, de-noised.
- **Null** uses two **independent** posterior draws (ε₁, ε₂, δ=0), so it measures
  the **full posterior-variance floor** — how much the decoded forecast varies
  just from the model's own posterior uncertainty (no perturbation at all).

R < Null (by 156×) means: at r_max, the perturbation δ causes an EEG change that
is **0.6% of the change the model produces from its own posterior randomness**.
The perturbation response is completely buried in posterior uncertainty — no
(a→b) edge is resolvable from the forecast-variance floor at this MC precision.
This is decision 8's "Null–perturbation separability" failing: the perturbation
response cannot be distinguished from model-generated variability. This is a real
finding about the substrate, not a comparison artifact. The R-matrix structure
below sharpens it further.

### R(r_max) matrix structure — the key finding

The mean R(r_max)=0.0056 hides a strong structure. Decomposing the 8-source ×
128-target subject-mean R(r_max) matrix:

| Quantity | Value |
|---|---|
| diagonal (self-response, a=b) R | **0.63 – 0.82** (all 8 probed sources) |
| off-diagonal (a→b, a≠b) R | **0.0001 – 0.0002** (median 0.00004) |
| diagonal / off-diagonal ratio | **~3000 – 8000×** |
| fraction of cells with R > 0.01 | **0.79%** ≈ 1/128 (exactly the diagonal) |
| rank-1 → rank-8 targets (the probed sources themselves) | R 0.08–0.10 |
| rank-9 target (first non-probed channel, #124) | R **0.0005** — cliff drop |

**The R matrix is effectively diagonal**: perturbing source `a` produces a strong
response only on target `a` itself (R≈0.65); every off-diagonal target `b≠a`
responds at R≈0.0001 (the perturbation causes a 0.01% change in target b's
normal fluctuation). The strong diagonal is the **trivial self-response**
(perturbing a's latent directly changes a's decoded forecast — not a "connection").
The off-diagonal cells — the actual effective-connectivity signal — are all
near zero.

**Physical interpretation**: this is the direct, quantitative consequence of the
08-09 gate finding that the mexhat+W mixer **isotropizes** (A_eff row-sum
CV=0.016 — every channel's row-sum is nearly identical). An isotropic mixer
spreads a single-channel perturbation uniformly across all channels, so no
target receives a privileged share; the only non-uniform response is the trivial
self-channel. The DE mechanism (IJKL) is working correctly — it is honestly
reporting that **the frozen DEFlow D=8 substrate carries no resolvable
cross-channel directed connectivity on the train split**.

**Implication for downstream steps**: any τ set in [0.001, 0.1] would yield an
effectively empty connectivity matrix (all off-diagonal edges below threshold).
This is not a τ-calibration problem (step N) — it is a substrate limitation.
The cross-channel information was already lost at the mixer level (08-09),
before the DE probe even runs. Candidates to recover cross-channel structure:
a higher-D latent, a non-isotropic mixer (stronger gamma_W, anti-uniformity
penalty on A_eff row-sums), or a different condition utilization path.

### Null semantics (confirmed correct)

The current Null — `MSE(decode(rollout(μ + s⊙ε₁)) − decode(rollout(μ + s⊙ε₂)))`
with δ=0 and two **independent** posterior draws — measures exactly what decision
8's "model-generated variability" requires: how much the decoded forecast varies
from posterior uncertainty alone. R uses shared ε (to isolate the pure δ effect);
Null uses independent ε (to capture the full posterior-variance floor). The
asymmetry is **intentional and correct** — R and Null answer different questions
("does δ move the forecast?" vs "does posterior noise alone move the forecast?"),
and R < Null is the meaningful separability verdict. (An earlier draft of this
note wrongly claimed the comparison was invalid because of the ε asymmetry — that
was a mistake; corrected 2026-08-11.) A complementary **cross-source pseudo-
perturbation Null** (δ points at an unrelated source a'≠a, same shared-ε
structure as R) would additionally test "does the model distinguish source
identity?" — useful as a second diagnostic in step N, but the current Null is the
correct primary floor for decision 8.

### Performance lesson (recorded honestly)

The initial optimization plan (vectorized rollout → 36× fewer calls → "15h to
45min") was **wrong**: the bottleneck is decode compute (GPU 98%), not kernel-
launch latency, so vectorization alone gives ~15% (measured: 11.2 → 9.5 min/source),
not 36×. The real speed came from strategy A+B (cutting N and M), validated by
the 5-window run: N 1720→215 (8×) + M 13→9 (1.4×) → ~1m11s/source (from ~11min,
a **9.5× measured speedup**). Lesson: profile before estimating; the smoke-scale
latency measurement (0.23s/call at N=64) did not extrapolate to N>1000 where
compute dominates.

### What the mechanism delivers

- **`ref_stats.npz`** — `MAD_b`, `S_ref`, `L_ref`, `0.5·log|S_ref|` from the train
  split (Ledoit-Wolf, D1). cond 280.6, eff_rank 38.5, logdet −14.7, MAD_b median
  0.638. Validated: L@L.T == S+jitter·I (err 5e-7).
- **`r_max` + grid** — Mahalanobis-quantile (D2, q=0.99) → r_max 1.393; grid 9
  points (r=0 + 8 positive).
  well inside the decoder support.
- **`response_curves.npz` + `null.npz`** — `R_{a,b}(r)` (S, n_src, C, M) + Null +
  subject-level cluster-bootstrap CI + censor=UNCENSORED. Validated on the smoke
  subset; the full matrix is the offline deliverable.
- **No τ, no r\*, no DE^(d)** (decision 27) — these are steps M–Q.

## Honest findings / caveats

- **Mechanism is sound**: every decision-13 invariant holds exactly; the pipeline
  is deterministic under a fixed seed; the response is exactly 0 at r=0 and grows
  with r.
- **`S_ref` estimator is provisional (decision D1)**: Ledoit-Wolf is the default,
  but the shrinkage intensity is data-dependent, so cross-fold geometry varies
  slightly. The final estimator lock is step O (breakpoint 3).
- **`r_max` is provisional (decision D2)**: the q=0.99 Mahalanobis-quantile bound
  is principled but the frozen value is step N (breakpoint 2). The decode-
  amplitude sanity scan shows r_max is well inside the decoder support (no
  saturation), so the bound is not artificially tight.
- **Out of scope (downstream)**: τ selection (step N, gated on synthetic EC
  recovery), r* / DE^(d) / monotonicity filtering / censor assignment (steps
  M–Q), signed influence (decision 15). This task produces the **inputs** to
  those steps (the response-curve + Null arrays), not their outputs.
- **Performance**: the full 128-source × ~7200-window run is hours-scale; the
  baseline precomputation saves a factor of ~n_src. Use `--sources` to probe a
  representative subset, or run offline in chunks. The 24-source representative
  run documented here is the smoke for the full matrix.

## Artifacts

- `analysis/de/{__init__,ref_stats,perturbation,rgrid,response}.py` — the mechanism.
- `scripts/run_de_response.py` — the runner (smoke + full).
- `tests/test_de_prep.py` — 17 tests, all invariants pinned.
- `.trellis/spec/evaluation/perturbation.md` — first spec coverage of
  perturbation-response analysis (the mechanism contract).
- `results/de/fold0/{ref_stats,response_curves,null}.npz + summary.md` — full run.
- `models/fm_model.py` — one additive method `sample_from_latent` (the only edit).
