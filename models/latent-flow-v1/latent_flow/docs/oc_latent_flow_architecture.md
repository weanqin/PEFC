# OC-LatentFlow: Architecture, Two-Stage Training, and DE Inference

> Authoritative description of the current active model
> (`models/oc_latent_flow.py`) and its connectivity-attribution path
> (`attribution/bisect_sigma.py`), derived from the code on `main`.
> All line references point to `models/oc_latent_flow.py` unless noted.
> Last updated 2026-08-06.

---

## 0. At a Glance

OC-LatentFlow is a **two-stage obs-conditioned latent flow-matching model** for
cross-subject EEG forecasting. The one-line paradigm:

```
Stage 1 (VAE):  x_obs -> encoder -> z_obs -> proj -> z_fc -> decoder -> x_hat
Stage 2 (Flow): enc(x_obs) = z_obs  --Euler(flow, 10 steps)-->  z_fc_hat -> decoder -> x_fc_hat
```

The defining design choice: **the flow transports the OBS latent to the fc latent
(`z_obs → z_fc`), not `N(0,I) → z_fc`**. This forces the encoder to encode
future-predictive information into `z_obs`, and makes the flow operate between
two *similar* latent distributions (measured at ~1/3 of the noise→fc distance;
see `domain.md` finding 2026-08-04).

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `x_obs` | `(B, 375, 128)` | 1.5 s observation (375 samples @ 250 Hz, 128 channels) |
| `x_fc` | `(B, 125, 128)` | 0.5 s forecast target = `x_target[:, 375:, :]` |
| `x_target` | `(B, 500, 128)` | full window (obs + fc) |
| `mu, logvar` | `(B, 128, 16, 64)` | VAE posterior mean / log-variance per element |
| `z_obs, z_fc` | `(B, 128, 16, 64)` | latent; `C=128` channels, `T_lat=16` latent time steps, `D=64` latent dim |
| `skips` | list of `(B*128, ch_i, T_down_i)` | U-Net encoder skip features (per-level, shallow→deep) |

> **Tensor convention** (project-wide): `(B, T, C)` for signals; the latent adds
> two axes `(B, C, T_lat, D)`. Shapes are documented inline at every non-obvious
> reshape.

---

## 1. Model Architecture

`OCLatentFlow` (`oc_latent_flow.py:847`) composes five modules. The backbone is
**pure 1D conv** (no Transformer — explicit project constraint).

### 1.1 ConvEncoder — per-channel temporal VAE encoder (`:69`)

```
x (B, T, C)
  -> permute/reshape to (B*C, 1, T)        # each EEG channel processed independently,
                                            #   shared conv weights across channels
  -> 4 levels of {Conv1d(k=7, stride=2) + GroupNorm + GELU}   # T downsampled
  -> AdaptiveAvgPool1d(T_lat=16)           # pin latent time steps exactly
  -> Conv1d(1×1) to 2*latent_dim           # split into mu + logvar
  -> reshape to (B, C, T_lat, 2*D)
  -> chunk(dim=-1) -> mu, logvar           # each (B, 128, 16, 64)
```

Key points:
- **Channel interaction is deliberately NOT modelled in the encoder** (docstring
  `:75-79`). The C channels are batched (`B*C`) and processed by shared temporal
  convs. All cross-channel mixing is delegated to the flow (downstream) — this
  preserves the per-channel latent structure that the connectivity goal needs.
- The VAE posterior is an **elementwise diagonal Gaussian**: every position
  `(c, t_lat, d)` has its own `mu` and `logvar`. `logvar` is clamped to `[-5, 2]`
  (`:134`).
- `return_skips=True` also returns per-level intermediate features for the
  decoder's U-Net path.

### 1.2 ProjHead — stage-1 bootstrap prediction head (`:144`)

A single `Linear(D, D)` applied per-channel to `z_obs` → `z_fc_pred`. Deliberately
simple: its only job is to give the stage-1 decoder a fc latent to reconstruct
against, pressuring the encoder to encode future info into `z_obs`.

Zero-initialized (weight + bias = 0) so `z_fc_pred ≈ z_obs` at init, giving the
decoder a sane starting point. **Replaced by the flow at inference.**

### 1.3 FlowVelocityNet — the velocity field `v(z_t, t, z_obs)` (`:519`)

This is the heart of the flow. It predicts the velocity of the ODE
`dz/dt = v(z, t, z_obs)` that transports `z_obs` (t=0) → `z_fc` (t=1).

Two interchangeable modes:

**Dense mode (legacy, `graph_mode=False`):**
```
[z_t, z_obs, t_emb] (B, C, T_lat, 3D) -> flatten (B, T_lat, C*D*3)
  -> Linear(C*D*3 -> hidden)               # isotropic cross-channel mix
  -> 6× {dilated Conv1d(k=3, dil ∈ {1,2,4,8,16,32}) + GN + GELU + 1×1 mix + GN + GELU, residual}
  -> Linear(hidden -> C*D) -> reshape (B, C, T_lat, D)
```

**Graph mode (`graph_mode=True`, the connectivity-active path):**
```
t_emb broadcast (B, C, T_lat, D)
  -> GraphChannelMixer([z_t, z_obs], t_emb)   # C preserved; K graph message-passing
  |                                            #   rounds over electrode adjacency
  -> (B, C, T_lat, hidden)
  -> dilated convs with C folded into batch   # strictly per-channel temporal,
  |                                            #   NO cross-channel mixing here
  -> per-channel output projection -> (B, C, T_lat, D)
```

The dilations `{1,2,4,8,16,32}` give a receptive field ≫ `T_lat=16`, so every
latent time step sees the whole latent sequence.

#### The channel mixer — where connectivity lives

The graph-mode mixer has two kernel variants:

- **`LearnableMexHatMixer`** (`:312`, default): a band-pass (Mexican-hat / Ricker
  or DoG) distance kernel gives **center excitation + surround inhibition**
  (lateral inhibition), preserving high-frequency channel structure instead of
  isotropizing it. On top of this *fixed* band-pass prior, a **learnable residual
  matrix `W`** lets the model express connections the distance kernel cannot
  (long-range functional links). This is the connectivity-inference site (the
  "mexhat breakthrough" — see `domain.md` finding 2026-06-30).
- **`GraphChannelMixer` / `GraphOutProj`** (`:173`, `:254`): Gaussian graph
  message passing; earlier variant, kept for old-checkpoint compatibility.

The **fixed adjacency `A_hat`** is built from electrode geometry
(`build_normalized_adj` / `data/modma/geometry.pt`) — it is a hard inductive bias
on *where* channels may connect, not on connection strength. Learned per-channel
Linear weights determine how strongly each node uses its neighbourhood.

### 1.4 ConvDecoder — U-Net skip decoder (`:689`) [full-window paradigm]

```
z_obs, z_fc (each B, C, T_lat, D)
  -> cat on T_lat axis -> (B, C, 2*T_lat, D) -> batch channels (B*C, D, 2*T_lat)
  -> 4 levels {ConvTranspose1d(stride=2) + GN + GELU, concat matching encoder skip, 1×1 fuse}
  -> 1×1 conv to 1 channel -> interpolate to T_full=500
  -> (B, T_full, C)
```

The **U-Net skip is mandatory, not optional**: feeding the decoder the TRUE
`z_fc = enc(x_fc)` without skips gave Pearson only 0.086 (std_ratio 0.30); with
skips the ideal path recovers high-frequency detail (z_obs PC1 15%→4.1%).
Reconstructs the full obs+fc window; the fc slice `x_hat[:, 375:, :]` is the
forecast.

### 1.5 ConvDecoderFc — fc-only decoder (`:774`) [fc-only paradigm, task 08-01]

```
z_fc (B, C, T_lat, D) -> batch channels (B*C, D, T_lat)
  -> 4 levels {ConvTranspose1d(stride=2) + GN + GELU}   # NO skip, NO z_obs
  -> 1×1 conv -> interpolate to forecast_len=125
  -> (B, 125, C)
```

Selected by `use_fc_decoder=True`. Generates **only** the fc segment from `z_fc`
alone. Motivation: the full-window ConvDecoder structurally drowns `z_fc` under
`z_obs + skip` (oracle `enc(x_fc)` gave fc Pearson only 0.032). Dropping obs
reconstruction entirely gives the decoder a single objective and single input.

### 1.6 Top-level orchestration

`OCLatentFlow.__init__` (`:863`) wires encoder + proj + flow + (decoder |
decoder_fc). Loss weights: `lambda_td/bd/corr/cons` (stage 1), `lambda_recon2/
corr2/cons2` (stage 2), `beta_kl_final=1e-4`. KL annealing via `set_kl_beta`
(called by the training script each epoch).

Checkpoint seam: `save_checkpoint` / `load_checkpoint` (`:1278`, `:1338`) handle
both `format_version=1` self-describing payloads (variant header → `_build_from_variant`
rebuilds topology) and legacy bare state-dicts (`_load_legacy` sniffs keys).
`weights_only=False` everywhere.

---

## 2. Stage 1 — VAE Training (`forward_stage1`, `:996`)

**Goal**: train the encoder + proj + decoder so that `z_obs` carries
future-predictive information and the decoder can reconstruct the fc segment.

**Encoder sees only `x_obs`** (decision B: the encoder never sees `x_fc` during
training — prediction is owned by proj/flow, not by the encoder cheating).

### Full-window paradigm (`use_fc_decoder=False`, default)

```
mu_obs, logvar_obs, skips = enc(x_obs, return_skips=True)
z_obs = reparam(mu_obs, logvar_obs)        # sampled posterior
z_fc_pred = proj(z_obs)                    # bootstrap fc latent
x_hat = decoder(z_obs, z_fc_pred, skips)   # reconstruct full obs+fc window
x_fc = x_target[:, 375:, :]                # ground truth fc segment
```

Loss:
```
loss = recon  (MAE: x_hat vs x_target, full window)
     + lambda_td   * temporal_diff(x_hat, x_target)   # Δ-time MAE
     + lambda_bd   * boundary(x_hat, x_target)        # obs→fc boundary continuity
     + current_beta * KL(mu_obs, logvar_obs)          # KL → N(0,I), annealed
     + lambda_corr * (1 - pearson_fc(x_hat_fc, x_fc)) # per-channel Pearson
     + lambda_cons * cons_fc(x_hat_fc, x_fc)          # mean/var consistency
```

### fc-only paradigm (`use_fc_decoder=True`, task 08-01)

```
mu_obs, logvar_obs = enc(x_obs)
z_obs = reparam(mu_obs, logvar_obs)
z_fc_pred = proj(z_obs)
x_fc_hat = decoder_fc(z_fc_pred)           # fc only, no z_obs, no skip
```

Loss drops obs recon / boundary (no obs output, no obs→fc boundary). `z_obs` has
no reconstruction anchor — **KL is the sole posterior constraint** (monitor PC1
for collapse). This is the path that produces `fold1_oc_latent_flow_vae_fconly.pt`.

### KL annealing
`beta_kl_final=1e-4`, annealed from 0 over epochs via `set_kl_beta`. Balances
VAE compression (recon ~0.7) against latent regularity for flow matching. The raw
KL diverges to ~90 on OC-LatentFlow — this is benign (see `domain.md` finding
"KL divergence is benign at beta_kl=1e-4").

---

## 3. Stage 2 — Flow Training (`forward_stage2`, `:1075`)

**Goal**: with encoder/proj/decoder **frozen**, train the flow to learn the
`z_obs → z_fc` transport.

```
mu_obs, logvar_obs, skips = enc(x_obs, return_skips=True)
z_obs = mu_obs                            # DETERMINISTIC at stage 2 (no reparam)
z_fc_target, _ = enc(x_fc)                # real fc latent, SAME frozen encoder on x_fc
                                          #   (decision-B fix: was proj(z_obs) → capped at proj's level)

# Flow matching on the z_obs -> z_fc_target line (OT-CFM straight path)
t ~ U[0,1],  t_e = t.view(B,1,1,1)
z_t   = (1 - t_e) * z_obs + t_e * z_fc_target    # linear interpolation
v_pred = flow(z_t, t, z_obs)                      # velocity regression
z_fc_hat = z_t + (1 - t_e) * v_pred               # single-step fc prediction
```

Decoder maps `z_fc_hat` back to fc signal; the **fc-signal losses drive training**:

```
loss = lambda_recon2 * recon2(MAE: x_fc_hat_fc vs x_fc)
     + lambda_corr2  * corr2(1 - pearson_fc)
     + lambda_cons2  * cons2(mean/var consistency)
     + current_beta  * KL(mu_obs, logvar_obs)      # no-op for grads when frozen
```

**Critical design points:**

1. **The flow target is the REAL `enc(x_fc)`, not `proj(z_obs)`.** The earlier
   "design B" used `proj(z_obs)` as the target — a zero-init linear head that
   capped the flow at proj's own Pearson (~0.20). Encoding the real `x_fc` with
   the same frozen encoder gives the flow a genuine latent distribution to
   transport to (`:1099-1108`).

2. **Single-step training, multi-step inference (train/inference gap).** Training
   uses one Euler step (`z_fc_hat = z_t + (1-t)*v_pred`); inference integrates
   10 steps. `v_pred` has no explicit target — its target is whatever latent
   makes the decoder's fc output correct (gradients flow through the frozen
   decoder).

3. **KL on z_obs is included but is a gradient no-op when frozen** (logvar is
   constant). It becomes active under joint training (`--unfreeze-vae`), where it
   prevents posterior drift. ⚠️ **Joint training currently leaks `x_fc` through
   `z_fc_target`** — see `domain.md` finding 2026-08 (task 08-01); do not trust
   joint-training metrics until fixed.

4. **Stage-2 is deterministic** (`z_obs = mu_obs`, no reparameterize). This is
   what makes the perturbation-response attribution clean (no forecast noise to
   confound the deltas).

### Inference — `sample()` (`:1156`)

```
mu_obs (+ skips) = enc(x_obs)
z_obs = mu_obs
if use_proj:                              # stage-1 bootstrap path
    z_fc_hat = proj(z_obs)                #   (eval a VAE before flow is trained)
else:                                     # production path
    z = z_obs
    for i in range(n_steps):              # n_steps=10
        t = i / n_steps
        v = flow(z, t, z_obs)
        z = z + (1/n_steps) * v           # Euler integration dz/dt = v
    z_fc_hat = z
x_hat = decoder(z_obs, z_fc_hat, skips)   # or decoder_fc(z_fc_hat) if fc-only
x_hat = nan_to_num(x_hat)                 # divergence guard
```

`use_proj=True` is required when evaluating a stage-1 (VAE-only) checkpoint —
the untrained flow would corrupt the latent. `use_proj=False` (default) is the
production stage-2 path.

---

## 4. DE Connectivity Inference (`attribution/bisect_sigma.py`)

**Goal**: infer a channel→channel connectivity matrix from the trained forecast
model by measuring how perturbing each source channel destabilises each target
channel's forecast.

### 4.1 The discard-entropy connectivity formula

```
connectivity[a, b] = -(log σ*[a,b] + 0.5·log(2πe))     (:36, :320)
```

where `σ*^a[b]` = the critical perturbation amplitude on source channel `a` that
just destabilises target `b`'s forecast (response reaches threshold `τ`). This is
the **negative differential entropy of a 1-D Gaussian** `N(0, (σ*)²)`:

- `σ*` small → a tiny perturbation on `a` destabilises `b` → **strong a→b**.
- `σ*` large → big perturbation needed → weak a→b (possibly negative value).

`_HALF_LOG_2PIE = 0.5·log(2πe)` is a constant (`:86`). Same formula as the legacy
JET/SGD path (`dropout_entropy.compute_connectivity`), kept identical so JET/OC
results are directly comparable.

### 4.2 The threshold τ — shared median (not per-target)

```
τ = alpha · median_b( MSE_b(σ_max) )      (:308)    DEFAULT_ALPHA = 0.5  (:105)
```

Every target gets the **same** bar: half of the *median* target's full-scale
(σ=σ_max) response. Why not per-target (`alpha·MSE_b@max`)? A few
hyper-responsive targets (MSE@max ~0.18 vs typical ~8e-5, a 2000× spread) got
enormous thresholds, were clamped to σ_max, and were judged *weak* despite
responding strongest — **inverting the signal** and producing a systematic
upper/lower-region heatmap split. The median is robust to that long tail;
response strength then flows directly into σ*.

`alpha=0.5` (not 1.0): 1.0 pushes σ*→σ_max for nearly all targets (no
discrimination); 0.5 places the bar in the mid-response range (connectivity ratio
front/back 2.2× → 0.97).

### 4.3 The algorithm — fixed-grid sweep + monotone walk (forward-only)

```
for source channel a in [0, C):
    draw a FIXED noise pattern xi ~ N(0,1) of shape (N, T_obs)  # seeded; reused across all σ
    x_fc_base = forecast_fc(model, x_obs)                        # unperturbed baseline
    mse_at_max = perturb_and_measure(σ_max)                      # for τ
    τ = alpha · median(mse_at_max)

    # Fixed grid sweep: n_grid=24 σ values, one forward each
    mse_curve = [ perturb_and_measure(σ) for σ in grid ]         # (n_grid, C)

    # Per-target monotone walk: MSE_b(σ) is monotone non-decreasing in σ,
    # so {σ : MSE_b(σ) ≤ τ_b} is a contiguous interval [0, σ*]; find its upper bound
    # by locating the last stable grid point and linearly interpolating the crossing.
    σ*[a, :] = walk_monotone_curve(mse_curve, grid, τ, σ_min, σ_max)

connectivity = compute_connectivity(σ* matrix)   # -(log σ* + 0.5 log(2πe))
```

**Why bisection (not SGD)?** `MSE_b(σ)` is monotone non-decreasing in σ (σ=0 →
MSE=0; larger σ → larger output deviation; the forecast map is continuous in its
input). So the stable set is a contiguous interval and bisection always
converges. It is **forward-only** (no gradients, no `_sample_impl`) — a 24-point
grid sweep evaluates every target's curve in 24 forwards. It also sidesteps the
JET stochasticity pitfall: OC stage-2 is deterministic (`z_obs = mu_obs`), so
perturbed-vs-baseline deltas are clean.

### 4.4 The perturbation primitive

```
x_pert[:, :, source_channel] = x_obs[:, :, source_channel] + σ · xi   (:186)
```

`xi` is drawn once per source from a seeded generator and **reused across every
σ** in that source's sweep — otherwise `MSE(σ)` would conflate amplitude and
noise-sample effects.

**Directional variant** (`mu_bias`, task 07-23): add a fixed-direction bias
`+ mu_bias[source_channel]` (not scaled by σ) to recover excitation/inhibition
sign. The site returns both `(mse, mean_diff)` so directional connectivity can be
inferred from the signed mean response, not just the magnitude.

### 4.5 ProbeSite — the perturbation-paradigm interface (`:421`)

`ProbeSite` encapsulates ONE perturbation paradigm + measurement site. The engine
(`attribution_engine`) is site-agnostic; it owns only the σ-search (grid sweep →
τ-median → monotone walk). Two concrete sites:

- **`SignalProbeSite`** — perturb `x_obs[:,:,a]` (signal space), measure on
  `x_fc` (full encoder→flow→decoder path).
- **`LatentProbeSite`** — perturb `z_obs[:,a,:,:]` (latent space), measure on
  `z_fc_hat` (encoder run once, decoder bypassed). Isolates the flow's intrinsic
  channel mixing from encoder/decoder confounds.

`perturb_and_measure(σ, source)` is the site's contract: apply a σ-amplitude
perturbation to `source`, forecast, measure, return per-target deviation. The
fixed-noise-pattern requirement is the site's responsibility.

### 4.6 Hyperparameters (defaults, `:126`)

| Key | Default | Meaning |
|-----|---------|---------|
| `alpha` | 0.5 | τ fraction of median max response |
| `sigma_min/max` | 0.01 / 0.3 | σ search range |
| `precision` | 1e-3 | (legacy; grid sweep supersedes bracket bisection) |
| `n_steps` | 10 | OC flow Euler steps (matches inference) |
| `n_grid` | 24 | σ grid points (one forward each) |
| `batch_size` | 32 | trial batch (RTX 5070 Ti 12GB sweet spot) |
| `amp_dtype` | bfloat16 | autocast dtype |
| `seed` | 42 | noise-pattern seed |

---

## 5. Key Constraints & Gotchas

- **No `torch.compile`** — broken on Blackwell sm_120 (see `training/numerics.md`).
- **No `conda run` on Windows** — gbk encoding issue; invoke `npi` python directly.
- **Stage-1 eval MUST use `sample(use_proj=True)`** — the default `sample()` walks
  the (untrained-at-stage-1) flow and corrupts the latent (bogus negative Pearson).
- **Checkpoint load is `weights_only=False`** everywhere; legacy bare state-dicts
  route through `_load_legacy` key-sniffing.
- **`σ*` boundary clamping preserves discard-entropy semantics**: a target already
  destabilised at σ_min gets `σ* = σ_min` (strongest possible connection); one
  never destabilised gets `σ* = σ_max`.
- **DE connectivity is bounded by the model's intrinsic channel sensitivity** —
  tuning attribution hyperparameters alone cannot break through it; the model
  itself must learn channel-specific dynamics (see `domain.md` 2026-06-18 finding).

---

## 6. Where Things Live

| Component | File:line |
|-----------|-----------|
| Model | `models/oc_latent_flow.py:847` |
| Encoder | `models/oc_latent_flow.py:69` |
| ProjHead (stage-1 bootstrap) | `models/oc_latent_flow.py:144` |
| FlowVelocityNet | `models/oc_latent_flow.py:519` |
| MexHat mixer (connectivity site) | `models/oc_latent_flow.py:312` |
| ConvDecoder (full-window) | `models/oc_latent_flow.py:689` |
| ConvDecoderFc (fc-only) | `models/oc_latent_flow.py:774` |
| `forward_stage1` | `models/oc_latent_flow.py:996` |
| `forward_stage2` | `models/oc_latent_flow.py:1075` |
| `sample` (inference) | `models/oc_latent_flow.py:1156` |
| Checkpoint save/load | `models/oc_latent_flow.py:1278` / `:1338` |
| DE engine | `attribution/bisect_sigma.py` |
| Connectivity formula | `attribution/bisect_sigma.py:317` |
| τ median threshold | `attribution/bisect_sigma.py:308` |
| Monotone walk core | `attribution/bisect_sigma.py:489` |
| ProbeSite interface | `attribution/bisect_sigma.py:421` |
