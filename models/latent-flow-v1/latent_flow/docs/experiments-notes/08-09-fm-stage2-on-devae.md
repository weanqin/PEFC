# Stage-2 Flow Matching on the DEVAE latent (mexhat+W mixer), steps E–H

> Task `08-09-fm-stage2-on-devae`. Implements experiment-plan steps E, F, G, H of
> `docs/OC_LatentFlow_DE_confirmed_decisions_summary.md` §2.2 — the Stage-2
> Flow-Matching arm of the DE-inference redesign.
>
> **Result type: mixed** — step G (condition utilization) PASSES decisively, but
> step F (cross-subject forecast) FAILS: the FM fits the training transition
> well yet does not generalize to held-out subjects. The qualification gate
> (step H) reflects this honestly.

## Setup

- **Frozen Stage-1 substrate**: DEVAE `D=8 / beta=0.01` (variant A),
  `checkpoints/de_vae/fold0_de_vae_D8_b0.01_mkl_variantA_best.pt`.
  Latent `(B, C=128, T_lat=16, D=8)`, per-channel `d=128`.
- **FM model** (`models/fm_model.py::DEFlow`): frozen DEVAE by composition +
  `FlowVelocityNet(graph_mode=True, graph_kernel="mexhat")`. The cross-channel
  mixer is the project's `LearnableMexHatMixer`
  (`A_eff = gamma_m·M(d) + gamma·W`, fixed Ricker band-pass + learnable directed
  residual `W`), imported as-is from `oc_latent_flow.py`.
- **Loss** (decision 1): explicit `L_FM = ‖v_θ(z_t,t,c_obs) − sg(z_fc−z_obs)‖²`
  PRIMARY (λ=1.0) + auxiliary recon/corr/cons (0.5/0.5/0.1). Shared-`ε` reparam
  (decision 2); deterministic condition `c_obs = μ_obs`.
- **Training**: `train_de_flow.py`, fold 0, L=128, 50 epochs, patience 15,
  bf16 AMP, grad-clip 1.0, Adam flow-only (451,993 params; VAE frozen
  488,529 params not in the optimizer). Trainable params == flow params (asserted).
- **No edit** to `de_vae.py` or `oc_latent_flow.py` (verified); `LearnableMexHatMixer`
  and `FlowVelocityNet` are imported, not copied.

## Step E — FM training (converged on train)

The flow learns the velocity target cleanly:

| epoch | train fm | val fm | val corr2 (≈1−Pearson) | ‖v_pred‖ | A_eff CV | \|W\| |
|---|---|---|---|---|---|---|
| 1 | 1.059 | 0.730 | 0.104 | 32.3 | 1.225 | 0.14 |
| 25 | 0.098 | 0.100 | 0.030 | 40.7 | 0.090 | 0.77 |
| 41 (best val) | 0.085 | **0.078** | **0.027** | 38.9 | 0.016 | 0.93 |
| 50 | 0.079 | 0.080 | 0.031 | 39.1 | 0.002 | 1.00 |

- **`v_norm` stays ~40** throughout — the flow does **not** collapse to a no-op
  (the OC failure signature from `spec/models/architecture.md` anti-patterns).
- **Train `corr2` → 0.027** ⇒ training fc Pearson ≈ 0.97. The FM reconstructs
  the fc segment from `z_obs` on the training subjects.
- **A_eff row-sum CV collapses 1.225 → 0.002** while `|W|` grows 0.14 → 1.00. The
  visualization (`scripts/plot_de_flow_aeff.py`,
  `results/de_flow/fold0/fold0_de_flow_aeff.png`) decomposes the mechanism: with
  `gamma_m=1.0, gamma_W=0.1` fixed, the final row-sums are M (mean −25.9, std 2.6)
  + γW (mean +6.7, std 2.3) → A_eff (mean −19.2, **std 0.31**). The positive γW
  row-sum offset partially cancels M's negative row-sum baseline, leaving every
  channel's row-sum at nearly the same value (CV 0.016) — i.e. the effective
  operator **isotropizes** (averages channels nearly uniformly), even though W's
  *off-diagonal* directed structure is still present (W is 30% negative, CV 0.34,
  with visible long-range links). The known mexhat failure mode (see `domain.md`
  "mexhat breakthrough" and the OC uniformity-penalty work); it does not block
  forecast quality but it **does** undermine the DE connectivity interpretation —
  the per-channel asymmetry the mixer is supposed to enforce is washed out. Fix
  levers: stronger `gamma_W`, an anti-uniformity penalty on A_eff row-sums, or
  normalizing W's row-sums.

Checkpoint: `checkpoints/de_flow/fold0_de_flow_best.pt`.

## Step F — Forecast evaluation (FAILS to generalize cross-subject)

`results/de_flow/fold0/fold0_de_flow_metrics_test.json`, held-out test (10 subj):

| source | MAE | Pearson | PSD-L1 | std_ratio | gate |
|---|---|---|---|---|---|
| **Oracle** `decode(encode(x_fc))` | 0.038 | **0.998** | 0.053 | 0.983 | PASS |
| **Endpoint** (1 Euler step) | 1.223 | **−0.040** | 0.380 | 0.980 | PASS |
| **Rollout** (10 Euler steps) | 0.946 | **0.004** | 0.840 | 0.454 | FAIL |

- **Oracle = 0.998**: the frozen DEVAE decoder can reconstruct the fc segment
  almost perfectly — the representation ceiling is high. So the failure is NOT
  a representation problem (decision 3 is satisfied).
- **Endpoint (1-step) Pearson = −0.04, Rollout (10-step) = 0.004**: the flow's
  forecast on held-out subjects is essentially zero-signal, despite training
  fc Pearson ≈ 0.97.
- **Endpoint ≈ Rollout**: the failure is NOT the OC train/inference step-count
  mismatch (endpoint, which mirrors the training single-step, is just as bad).
  It is **cross-subject over-fitting** — the FM memorizes the training subjects'
  transitions and does not transfer. This is consistent with the gate's 08-09
  finding that the direct-`z_fc` cross-subject signal is weak (R²≈0.10).
- **No latent collapse**: PC1 = 6.5% (< 50%), eff-dim 78.5/16384.
- **Verdict (decision 3 check)**: Oracle ≥ rollout — holds (the ceiling is
  respected). But the gap is the entire signal.

## Step G — Condition utilization (PASS, decisively)

`results/de_flow/fold0/fold0_de_flow_condition_test.json`, fixed `z_t`, t=0.5,
vary only `c_obs`:

| condition | L_FM | rollout Pearson | rollout MAE | Δfm vs correct |
|---|---|---|---|---|
| correct (`μ_obs`) | **0.026** | 0.004 | 0.935 | — |
| batch-shuffled | 10.746 | −0.034 | 1.568 | **+10.720** |
| zeroed | 5.324 | −0.039 | 1.380 | **+5.298** |
| cross-subject | 11.046 | −0.035 | 1.581 | **+11.020** |

**Verdict: PASS — the model uses the condition** (decision 7/8 satisfied). The
correct-condition velocity error is **~400× smaller** than any wrong condition.
So the FM is mechanistically sound (it reads `z_obs` to predict the transition);
the problem is that the *transition itself* does not generalize across subjects.

## Step H — Qualification gate (T-004 realized; verdict reflects F)

The existing `analysis/fm_gate/` (T-001/T-002/T-002D from 08-08/08-09) is reused
verbatim; this task **realizes T-004** (FM Skill, previously a stub) in
`analysis/fm_gate/fm_skill.py` and threads it into `verdict.py`. T-004 measures
whether the trained FM beats the zero-velocity baseline on the SAME
`fold0_pairs.npz` substrate the other probes used (apples-to-apples).

**T-004 result** (`results/fm_gate/fold0_t004.json`, val substrate):

| metric | value | CI (1000 subj-cluster resamples) |
|---|---|---|
| `Skill_FM` (channel median) | **0.981** | [0.978, 0.983] |
| status | **LEARNED** | (ci_low > 0) |
| `L_FM` (channel median) | 0.058 | — |
| `L_zero` (channel median) | 3.015 | — |
| cheat-check | consumes `z_t`=True, consumes `c_obs`=True, **cheat_detected=False** | — |

So on the **val** substrate the FM is LEARNED by a huge margin: it recovers 98%
of the velocity signal relative to the zero-velocity baseline, and it genuinely
consumes both inputs (no shortcut).

**Re-issued verdict** (`results/fm_gate/fold0_verdict_v3.json`; v1/v2 untouched):

- `final_verdict: INCONCLUSIVE`, `routing: COLLECT_MORE_EVIDENCE_OR_LONGER_DATA`.
- `t004_status: LEARNED` (was `FM_NOT_TRAINED` in v2).
- The T-004 early-reject branch did **not** fire (`Skill_FM` ci_low=0.978 ≫ 0),
  so it does not override the verdict.
- Case III PASS still not met: `R²_direct_mlp` ci_low=0.028 < `MIN_R2=0.05`;
  `gain_cross` ci_low=−0.096 < `MIN_GAIN=0.02`.

**This is the central honest finding.** T-004 (LEARNED) and T-002/F (weak
cross-subject) are not contradictory — they measure different things:
- T-004 = "can the FM regress the *velocity target* on the training-fold
  subjects?" → **yes** (Skill 0.98).
- T-002 `R²_direct_mlp` = "does `z_obs` predict `z_fc` **across subjects**?" →
  barely (R² 0.028–0.10), the same wall the 08-09 gate found.
- Step F = "does the FM *forecast* generalize to held-out **test** subjects?"
  → **no** (test Pearson 0.004).

So the FM is a well-trained velocity regressor that does not transfer. The gate
surfaces exactly this nuance — its purpose was to prevent spending GPU on an
unlearnable transition, and "learned but non-generalizing" is the honest read.

## Interpretation (honest)

1. **The FM works as designed mechanically.** It learns an explicit velocity
   field on the frozen DEVAE latent, uses the observation condition (step G
   passes by 400×), and does not collapse (v_norm stable). Decision 1 (explicit
   FM as primary loss) is validated as a training recipe.
2. **Cross-subject generalization is the bottleneck.** Training fc Pearson 0.97
   vs test 0.004 is a ~0.97 train→test gap — the hallmark of memorization, not
   learning. This is the same wall the 08-09 gate measured (direct-`z_fc` R²≈0.10):
   the frozen D=8/beta=0.01 latent carries only a weak cross-subject predictive
   signal, and a 50-epoch FM can fit the training subjects without extracting
   the transferable part.
3. **The mexhat prior collapsed.** A_eff CV → 0.002 means the band-pass distance
   kernel was overwhelmed by the learnable `W`. The "mexhat + W" hybrid was
   supposed to keep structural channel priors; in this run `W` won completely.
   This is a connectivity-inference concern (the mixer is the future DE site),
   independent of the forecast result.
4. **Implication for the plan.** Steps I–S (DE perturbation/τ/connectivity) are
   NOT blocked by a methodology gap — they are blocked by the cross-subject
   forecast not generalizing. Before DE, the substrate must change: candidates
   are a higher-D / differently-regularized DEVAE (the D=16/32 checkpoints
   exist), L=256 (decision 4), anti-uniformity regularization on A_eff, or a
   cross-subject-generalization intervention on the FM itself.

## Artifacts

- Model: `models/fm_model.py` (`DEFlow`).
- Scripts: `scripts/{train,eval,probe_de_flow_condition,run_fm_skill,shape_check_de_flow}.py`.
- Analysis: `analysis/fm_gate/fm_skill.py` (T-004).
- Tests: `tests/test_de_flow.py` (12), `tests/test_fm_gate.py` (+3 T-004).
- Checkpoints/results: `checkpoints/de_flow/fold0_de_flow_best.pt`,
  `logs/de_flow/fold0_de_flow_history.json`,
  `results/de_flow/fold0/{metrics,condition,summary}_test.{json,md}`,
  `results/fm_gate/{fold0_t004.json, fold0_verdict_v3.json}`.

## Commands (reproduce, from `D:\102-npi`, env `npi`)

```bash
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.train_de_flow --fold 0 --segment-len 128 --epochs 50 --patience 15
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.eval_de_flow --fold 0 --split test
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.probe_de_flow_condition --fold 0 --split test
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_fm_skill --fold 0 --split val --n-resamples 200
D:/program/anaconda/envs/npi/python.exe -u -m latent_flow.scripts.run_fm_gate_controller --fold 0 --suffix _v3 --t004-json results/fm_gate/fold0_t004.json
```
