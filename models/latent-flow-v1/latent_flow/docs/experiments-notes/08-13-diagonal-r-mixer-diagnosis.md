---
date: 2026-08-13
name: diagonal-r-mixer-diagnosis
aliases:
  - mixer probe
  - Experiment 1
  - mixer capacity probe
task: 08-13-diagonal-r-mixer-diagnosis
type: diagnosis (synthetic-teacher training + evaluation probe)
dataset: modma
fold: 0
status: 完成 (verdict AMBIGUOUS — borderline A)
tags:
  - experiment
  - diagnosis
  - modma
  - mixer-probe
  - cross-channel
  - ambiguous-result
checkpoint: null        # no new checkpoint; real-FM mixer read read-only from fold0_de_flow_best.pt
variant: null
related_spec: "[[perturbation]]"
---

# Diagonal-R diagnosis, Experiment 1 — can the Mixer do cross-channel mixing?

> Task `08-13-diagonal-r-mixer-diagnosis` (direct follow-up to
> [[08-10-de-inference-prep-ijkl]]). The archived 08-10 finding is that the DE
> response matrix $R_{a,b}(r_{\max})$ is **effectively diagonal** — a
> source-channel perturbation never propagates to other channels' forecasts.
> This experiment isolates the prime suspect (the flow's cross-channel mixer,
> `LearnableMexHatMixer`, $A_{\mathrm{eff}} = \gamma_m M(d) + \gamma W$) from the
> real Stage-2 objective via a **synthetic teacher with a known answer**.
>
> **Result type: mixed / nuanced.** Pre-registered classifier verdict:
> **AMBIGUOUS (borderline Case A)** — $M2{-}M1 = +0.191$ vs the pre-registered
> "$\ll$" threshold $0.20$ (missed by $0.009$). Triangulated across the R5
> internals, however, the outcome is sharp: **H2 ruled out, "module cannot
> learn" ruled out, the W-starvation hypothesis FALSIFIED** — the diagonal-R
> cause is localized **away from the mixer**, toward the real-FM objective /
> perturbation-transmission layer.

> [!summary] One-line conclusion
> The mixer's cross-channel capacity is largely fine (M2 $\gamma_m{=}0$ and M3 dense both reach $R^2 \approx 0.97{-}0.99$ on the anti-geometric teacher, and the real-FM-trained $W$ grew to off-diag $\|W\|_F = 143.95$, ~3× the synthetic fit); M1 (MexHat prior on) is impaired-but-climbing at long range (0.797 vs 0.988), so H1 is softened, not confirmed — the diagonal-R failure is **not** a mixer-capacity problem.

## Motivation and hypotheses

From [[08-10-de-inference-prep-ijkl]] (commit `045a05b`): on the frozen DEFlow
(fold 0, D8 VAE), perturbing source $a$ moves only channel $a$'s forecast
(diagonal/off-diagonal response ratio ~3000–8000×). Candidate explanations:

- **H1** — the Mexican-hat distance prior is the bottleneck (good local
  geometry, suppresses long-range / non-geometric links).
- **H2** — the graph / message-passing implementation is broken
  (normalization, residual scaling, rounds, aggregation, $W$
  parameterization, gradient flow).
- **Case C** — the mixer is fine; the real Stage-2 objective (or the VAE) does
  not drive it to learn cross-channel structure → Experiments 2+.

## Design (controlled comparison at maximum fidelity)

- **Shell**: all four variants are the **real**
  `FlowVelocityNet(graph_mode=True, graph_kernel="mexhat")` — the diagnosed
  model's velocity net verbatim; **only `.graph_mixer` differs** (decision D1).
  The dilated 1-D convs are strictly per-channel in graph mode ($C$ folded into
  batch), so cross-channel information can flow only through the mixer's
  $A @ x$ and the comparison is controlled. No file in `models/` was edited
  (`git diff -- models/` empty; M0/M3 are drop-ins assigned after
  construction).
- **Variants**: M0 `SelfOnlyMixer` ($A = I$, negative control) · M1 the
  **unmodified** real `LearnableMexHatMixer` · M2 real mixer with
  `mexhat_gamma_m = 0` (isolates H1) · M3 `DenseChannelMixer` (one round
  $W_{\mathrm{dense}}$ $C{\times}C$, diag masked — positive control, isolates H2).
- **Teachers** $A^{\mathrm{true}} \in \mathbb{R}^{128 \times 128}$, $\mathrm{diag}=0$:
  **Teacher-L** (local): $+1$ on the $k{=}8$ nearest electrodes (Euclidean 3D,
  `load_geometry("modma")`) — MexHat-friendly. **Teacher-R** (far,
  anti-geometric): $\pm 1$ seeded signs on the $k{=}8$ **farthest** electrodes —
  deliberately violates the distance prior. Target
  $Y_b = \sum_a A^{\mathrm{true}}_{ba} X_a$ (channel-axis mix; $A[i,j]$ = channel
  $j$ drives $i$, matching the mixer's `x_perm @ A.t()` convention — pinned by
  test).
- **Input / task**: $X = \mu_{\mathrm{obs}}$ from the frozen substrate
  `results/fm_gate/fold0_pairs.npz` (`z_obs` (8881, 128, 128) reshaped to
  $(N, C{=}128, T_{\mathrm{lat}}{=}16, D{=}8)$; train 6394 / val 814, 5 val
  subjects; no VAE loaded — $\mu$ is pre-extracted, decision D2).
  $v = \mathrm{vel\_net}(z_t{=}X,\ t{=}0,\ z_{\mathrm{obs}}{=}X)$ fit to
  $Y = A^{\mathrm{true}} X$ with MSE; pooled $R^2$ on val; gate $R^2 > 0.95$.
  Cross-channel transmission is certified by **gain over M0**, not raw $R^2$.

## Finding 1 — Teacher-L is non-discriminative (recorded finding, not a failure)

The $\mu_{\mathrm{obs}}$ substrate is strongly spatially autocorrelated
(measured on val): mean $\mathrm{corr}(X_a, X_b)$ = **0.880** for near pairs vs
0.089 far, 0.402 all off-diagonal. A self-only scalar (elementwise linear) fit
on train reaches **0.934** on Teacher-L (vs **0.031** on Teacher-R) — the
mean-of-8-neighbours target is ~93% predictable from a channel's own latent, so
**local cross-channel transmission is indistinguishable from self on this
substrate**. Teacher-L was therefore demoted to auxiliary; the verdict is
driven by Teacher-R (far), whose self-only floor is ~0.03.

Consistent with the floor, the Teacher-L arm (prior run, 3000 steps, batch 128,
seed 0, same substrate/seed): M0 = 0.9334 (≈ the 0.934 floor), M1 = 0.9943,
M2 = 0.99987. M3-local was never trained (aux arm left incomplete by design).

## Verdict table — Teacher-R (registered run: 1500 steps, batch 64, seed 0)

| mixer | $R^2_{\mathrm{synth}}$(val) | gain over M0 | med per-channel $R^2$ | gate | final loss |
|---|---|---|---|---|---|
| M0 SelfOnly | 0.0092 | +0.000 | −0.127 | fail | 15.45 |
| M1 MexHat+W (real, unmodified) | 0.7968 | +0.788 | 0.729 | fail | 3.144 |
| M2 $\gamma_m{=}0$ (W only) | 0.9880 | +0.979 | 0.986 | **PASS** | 0.110 |
| M3 Dense | 0.9735 | +0.964 | 0.975 | **PASS** | 0.280 |

Params: M0 435,608 (mixer 36,352) · M1/M2 451,993 (mixer 52,737) · M3 435,480
(mixer 36,224). Per-variant train time ~173–185 s.

> [!example] Key numbers
> Gaps on Teacher-R: $M2{-}M1 = +0.1912$ (pre-registered "$\ll$" = gap > 0.20 — **missed by 0.009**); $M3{-}M2 = -0.0144$ ($\approx$). Rule booleans: `M1≪M2` false, `M1≈M2` false (tolerance 0.10), `M2≈M3` true, all of M1/M2/M3 "clearly high" (gain > 0.3). Neither Case A nor Case C fires → **AMBIGUOUS, borderline A**. M1's final loss is ~29× M2's; M1 climbed 0.296 (200-step smoke) → 0.797 (1500 steps) — **impaired, not blocked**.

## M1 internals — the W-starvation hypothesis is FALSIFIED

Pre-experiment hypothesis (PRD R5): the real FM objective fails to drive $W$
("γ≈0 / ‖W_off‖≈0 under real FM"). The measurements contradict it:

| quantity | synthetic (Teacher-R fit) | real-FM (`fold0_de_flow_best.pt`) |
|---|---|---|
| off-diag $\|W\|_F$ | 48.94 | **143.95** |
| $\rho_W = \|\gamma W\|_F / \|\gamma_m M\|_F$ | 0.093 | **0.308** |
| $W$ off-diag sparsity | 1.8e-4 | **0.0 (dense)** |
| $\sigma$ (MexHat width) | 0.125 | 0.070 |
| $\gamma_m$, $\gamma$ | 1.0, 0.1 | 1.0, 0.1 |

The real Stage-2 objective drove $W$ to grow **~3× more** than the pure
cross-channel teacher did (48.9 → 143.9, from the same zero/$W$-init), with
$\rho_W$ 0.31 vs 0.09. Under the synthetic far teacher, M1's zero-init $W$
also grew to off-diag norm 48.9 and $\sigma$ widened 0.070 → 0.125 (straining
to reach far electrodes). The mixer **can** learn cross-channel structure, and
under the real FM objective it **did**.

## Adjudication

- **H2 (graph/message-passing broken): RULED OUT** — M2 ≈ M3 near ceiling
  (0.988 vs 0.974); the scaffold learns a cross-channel mix as well as a free
  dense matrix.
- **"Module cannot learn cross-channel": RULED OUT** — synthetic-M1's zero-init
  $W$ grew (off-diag 48.9), and the real-FM $W$ grew ~3× further (143.95, dense).
- **H1 (MexHat prior suppresses long-range): SOFTENED, not confirmed** — M1 is
  impaired at long range (0.797 vs 0.988) and its $\sigma$ strains to widen,
  but it keeps climbing with training; the formal classifier stays AMBIGUOUS
  because the $M2{-}M1$ gap sits 0.009 below the pre-registered threshold.
- **Conclusion**: mixer cross-channel capacity is largely fine. The diagonal-R
  problem is localized **away from the mixer**, toward the real-FM objective /
  perturbation-transmission layer — e.g. the real objective teaches $W$
  *forecast-relevant* cross-channel structure that does not transmit
  $S_{\mathrm{ref}}$-shaped perturbations $\delta = r \cdot L_{\mathrm{ref}} \xi$
  to other channels' forecasts; or the per-channel decoder absorbs cross-channel
  latent shifts. → **Experiment 2** (real-objective / perturbation-transmission
  probe), gated on exactly this verdict.

## Honest caveats / deviations

- **AMBIGUOUS is recorded as AMBIGUOUS** — not upgraded to A. The $M2{-}M1$
  gap (0.191) sits just under the pre-registered 0.20 "≪" threshold.
- **M1 may be under-converged** at 1500 steps (final loss 29× M2's; still
  descending). A 3000-step M1-only far tiebreaker was **proposed but NOT run**
  (user decision to proceed to wrap-up) — this is the open follow-up that would
  firm up AMBIGUOUS → A (or confirm C-side).
- **M3 on Teacher-L never trained** (aux arm incomplete by design; Teacher-L is
  non-discriminative anyway, floor 0.934).
- **Environment**: batch-128 **far** arm CUDA-faulted 4/4 (illegal memory
  access / `CUBLAS_STATUS_INTERNAL_ERROR`) on the 12 GB shared laptop GPU
  (torch 2.12.0.dev+cu128, sm_120); batch 64 was stable 4/4 — the registered
  run used batch 64. The 3000-step Teacher-L aux arm ran at batch 128 without
  faults. No `torch.compile`, AMP off (determinism).
- **Tests / blast radius**: `python -m pytest latent_flow/tests/` — 171 passed,
  3 skipped; `models/` untouched (`git diff` empty at HEAD `f247bf5`). New
  files are additive: `analysis/de/mixer_probe.py`,
  `scripts/run_mixer_probe.py`, `tests/test_mixer_probe.py`,
  `results/de/mixer_probe/`.

## Reproducibility

Run from the **parent** of `latent_flow/`, env `npi` (invoke the env's python
directly — no `conda run` on Windows, gbk crash):

```bash
# Gate A (unit tests: teacher construction, A@X convention, drop-in interface,
# M1-verbatim, M0 negative control, determinism)
python -m latent_flow.scripts.run_mixer_probe --fold 0 --smoke          # Gate B (200 steps, batch 64)

# Registered Teacher-R run (writes results/de/mixer_probe/)
python -m latent_flow.scripts.run_mixer_probe --fold 0 --teachers far \
    --steps 1500 --batch-size 64 --seed 0
# (remaining defaults: k-near 8, k-far 8, lr 1e-3, hidden-ch 128, k-layers 2)

# Teacher-L aux arm (prior run, provenance stamped in verdict.json)
python -m latent_flow.scripts.run_mixer_probe --fold 0 --teachers local \
    --steps 3000 --batch-size 128 --seed 0

python -m pytest latent_flow/tests/ -q
```

Seed 0 fixes `torch.manual_seed` + `np.random.default_rng` + teacher-sign RNG +
probe init (stamped in `verdict.json`). Substrate
`results/fm_gate/fold0_pairs.npz`; real-FM mixer read read-only from
`checkpoints/de_flow/fold0_de_flow_best.pt`.

## Next steps

- **Experiment 2 — real-objective / perturbation-transmission probe** (the
  gated next step): why does a large, dense, real-FM-trained $W$ still yield a
  diagonal $R$? Probe whether $S_{\mathrm{ref}}$-shaped $\delta$ perturbations
  transmit through the trained mixer to other channels' **forecast** outputs,
  and whether the per-channel decoder absorbs cross-channel latent shifts.
- **Optional tiebreaker**: 3000-step M1-only far run (resolve AMBIGUOUS vs firm
  H1).
- Spec updated: `.trellis/spec/evaluation/perturbation.md` § "Mixer capacity
  probe" documents the synthetic-teacher contract incl. the
  autocorrelation-floor lesson (probe with far/anti-geometric targets, not
  local ones, on autocorrelated substrates).

## Artifacts

| type | path |
|---|---|
| Verdict (machine-readable) | `results/de/mixer_probe/verdict.json` |
| Verdict (human) | `results/de/mixer_probe/summary.md` |
| Teacher-L aux arm (3000/128, incomplete) | `results/de/mixer_probe/teacher_local_aux.json` |
| M1 internals (synthetic vs real-FM) | `results/de/mixer_probe/m1_internals.npz` |
| Loss curves (far × M0–M3) | `results/de/mixer_probe/loss_curves.npz` |
| Probe module | `analysis/de/mixer_probe.py` |
| Runner | `scripts/run_mixer_probe.py` |
| Tests | `tests/test_mixer_probe.py` |

## Related notes

- Predecessor / motivation: [[08-10-de-inference-prep-ijkl]] (diagonal-R finding)
- Cross-channel learnability context: [[08-09-fm-gate-rework]] (T-002D), [[08-09-fm-stage2-on-devae]]
- Spec: [[perturbation]] (mixer capacity probe contract)
