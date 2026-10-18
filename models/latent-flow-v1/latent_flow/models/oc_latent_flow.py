"""Obs-Conditioned Latent Flow (OC-LatentFlow) for cross-subject EEG forecasting.

Task 06-23-new-latent-flow-arch. Design lives at
``.trellis/tasks/06-23-new-latent-flow-arch/design.md``.

Paradigm
--------
Two-stage obs-conditioned latent flow, inspired by PV-VAE (arXiv 2605.02134):

Stage 1 (VAE, encoder sees only obs):
    x_obs -> encoder -> z_obs (B, C, T_lat, D)
    z_fc = proj(z_obs)                    # prediction head, bootstrap
    decoder([z_obs, z_fc]) -> x_hat       # reconstruct obs + fc full window
    loss = recon(MAE) + temporal_diff + boundary + KL(z_obs) + corr_fc + cons_fc

Stage 2 (Flow, encoder/proj/decoder frozen, gradients flow through decoder):
    z_fc_target = enc(x_fc)               # real fc latent (same frozen encoder)
    flow learns z_obs -> z_fc transport (NOT N(0,I) -> z_fc)
    loss = end-to-end fc signal loss ONLY (no flow/proj distribution loss);
           the flow is optimized purely by gradients backpropagated through
           the frozen decoder from the fc-signal objectives.

Inference:
    x_obs -> encoder -> z_obs -> flow (Euler) -> z_fc_hat
    decoder([z_obs, z_fc_hat]) -> x_hat (obs+fc); fc slice is the forecast.

Key differences from the original latent_flow.py (which hit encoder collapse):
- per-channel temporal latent (B, C, T_lat, D) — time axis preserved (no global pool)
- decoder upsamples from temporal latent — fc waveform generated, not inherited
- flow starts from z_obs (not noise), forcing the encoder to encode the future
- encoder never sees fc (decision B); proj + flow own the prediction
"""

from __future__ import annotations

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Time embedding (flow matching timestep)
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for the flow-matching timestep t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=device) * (-torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        emb = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# =============================================================================
# Conv encoder — per-channel temporal downsampling
# =============================================================================

class ConvEncoder(nn.Module):
    """Per-channel 1D temporal conv encoder.

    Input  : (B, T, C)   e.g. (B, 375, 128) for obs or (B, 125, 128) for fc
    Output : mu, logvar  each (B, C, T_lat, D)

    The conv weights are shared across the C channels: the signal is treated
    as C independent 1D sequences that share one temporal conv stack. Channel
    interaction is intentionally NOT modelled inside the encoder (the flow's
    1x1 convs own channel mixing downstream), preserving the per-channel
    latent structure required by the connectivity goal.

    T is downsampled to T_lat via stride-2 convs. Because obs (375) and fc
    (125) differ, the same conv stack is applied but with a fixed total
    downsampling factor; an adaptive pool at the end pins T_lat exactly.
    """

    def __init__(self, n_channels: int = 128, latent_dim: int = 64,
                 t_lat: int = 16, base_ch: int = 32, n_levels: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.t_lat = t_lat

        # Per-channel temporal conv stack. Input has 1 channel (each EEG
        # channel processed independently); base_ch is the feature width.
        blocks = []
        in_ch = 1
        for i in range(n_levels):
            out_ch = base_ch * (2 ** i)
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.temporal_blocks = nn.ModuleList(blocks)

        self.flatten_ch = in_ch  # base_ch * 2^(n_levels-1)
        # Pin T to T_lat with an adaptive avg pool, then project to 2*latent_dim (mu+logvar)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(t_lat)
        self.to_latent = nn.Conv1d(self.flatten_ch, latent_dim * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, return_skips: bool = False):
        # x: (B, T, C) -> treat each channel as an independent 1D sequence
        B, T, C = x.shape
        assert C == self.n_channels, f"expected {self.n_channels} channels, got {C}"

        # (B, T, C) -> (B*C, T, 1): batch the channels so the shared temporal
        # conv processes every channel with the same weights.
        h = x.permute(0, 2, 1).reshape(B * C, T, 1).permute(0, 2, 1)  # (B*C, 1, T)

        skips = []
        for block in self.temporal_blocks:
            h = block(h)
            if return_skips:
                skips.append(h)  # (B*C, ch_i, T_down_i) per level

        # h: (B*C, flatten_ch, T_down) -> pin to T_lat
        h = self.adaptive_pool(h)                    # (B*C, flatten_ch, T_lat)
        h = self.to_latent(h)                         # (B*C, 2*latent_dim, T_lat)

        # Reshape back to (B, C, T_lat, 2*latent_dim)
        h = h.permute(0, 2, 1).reshape(B, C, self.t_lat, self.latent_dim * 2)
        mu, logvar = h.chunk(2, dim=-1)              # each (B, C, T_lat, D)
        logvar = torch.clamp(logvar, -5.0, 2.0)
        if return_skips:
            return mu, logvar, skips  # skips: list of (B*C, ch_i, T_down_i)
        return mu, logvar


# =============================================================================
# Prediction head — z_fc = proj(z_obs) bootstrap (decision B)
# =============================================================================

class ProjHead(nn.Module):
    """Predict fc latent from obs latent (stage-1 bootstrap; replaced by flow at inference).

    Input  : z_obs (B, C, T_lat, D)
    Output : z_fc_pred (B, C, T_lat, D)

    A single linear layer on the feature dim. This is deliberately simple: its
    only job is to give the stage-1 decoder a fc latent to reconstruct against
    so the encoder is pressured to encode future-predictive information into
    z_obs. The stage-2 flow (dilated conv, much larger capacity) learns the
    real z_obs -> z_fc transport and replaces proj at inference.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)
        # Zero-init so proj starts as identity-ish (z_fc_pred ≈ z_obs at init),
        # giving the decoder a sane starting point.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z_obs: torch.Tensor) -> torch.Tensor:
        return self.proj(z_obs)


# =============================================================================
# Flow velocity network — dilated 1D conv, z_obs -> z_fc transport
# =============================================================================

class GraphChannelMixer(nn.Module):
    """Geometry-conditioned channel mixer — R2 graph-prior injection (task 06-29).

    Replaces the dense ``input_proj: Linear(C*D*3 -> hidden)`` that flattened and
    isotropically mixed all channels (the R1-confirmed cross-channel suppressor).
    Instead the channel axis C is preserved throughout, and the ONLY cross-channel
    interaction is K layers of graph message passing over the EEG electrode
    adjacency (from geometry.pt). This concentrates all connectivity-relevant
    mixing into one interpretable, geometry-prior-conditioned site.

    Data flow (C preserved, T_lat preserved):
        [z_t, z_obs] (B, C, T_lat, 2D)  -- cat on D (condition z_obs kept per-ch)
          -> per-channel Linear(2D -> hidden)              (B, C, T_lat, hidden)
          + time embedding (broadcast, per-channel)         (B, C, T_lat, hidden)
          -> K x { A_hat @ X  (graph message pass over C)
                   per-channel Linear(hidden->hidden) + GELU }
          -> (B, C, T_lat, hidden)

    The adjacency A_hat is a FIXED buffer (from build_normalized_adj), NOT a
    learned parameter — the graph prior is a hard inductive bias on WHERE
    channels may connect, not on connection strength. Learned per-channel
    Linear weights determine how strongly each node uses its neighbourhood.

    The effective directed connectivity is read from this layer's K message-passing
    Linear weights (R4 clean path), so it is the connectivity-inference site.
    """

    def __init__(self, n_channels: int, latent_dim: int, hidden_ch: int,
                 k_layers: int = 2, a_hat: torch.Tensor = None):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.k_layers = k_layers

        # Per-channel input projection (NO cross-channel mixing here).
        # Input per channel = [z_t(D), z_obs(D)] = 2D features.
        self.in_proj = nn.Linear(latent_dim * 2, hidden_ch)

        # Time embedding is in latent_dim (D) space; project it to hidden_ch so
        # it can be added to the per-channel hidden representation. Per-channel
        # (weight-shared), broadcast across C and T_lat — a global condition.
        self.t_proj = nn.Linear(latent_dim, hidden_ch)

        # K graph message-passing layers. Each: aggregate neighbours via A_hat,
        # then a learned per-channel transform. Stacking learns multi-hop reach.
        self.msg_layers = nn.ModuleList(
            [nn.Linear(hidden_ch, hidden_ch) for _ in range(k_layers)])

        if a_hat is not None:
            self.register_buffer("a_hat", a_hat.float(), persistent=True)
        else:
            self.register_buffer("a_hat", torch.eye(n_channels), persistent=True)

    def set_adjacency(self, a_hat: torch.Tensor):
        """Replace the fixed adjacency buffer (e.g. after device move)."""
        self.a_hat = a_hat.float().to(self.a_hat.device)

    def forward(self, z_t: torch.Tensor, z_obs: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        """z_t, z_obs: (B, C, T_lat, D); t_emb: (B, C, T_lat, D).
        Returns (B, C, T_lat, hidden_ch)."""
        # Per-channel projection of [z_t, z_obs].
        x = torch.cat([z_t, z_obs], dim=-1)        # (B, C, T_lat, 2D)
        x = self.in_proj(x)                          # (B, C, T_lat, hidden)
        # Add the time embedding projected to hidden (broadcast condition, no
        # cross-channel mixing).
        x = x + self.t_proj(t_emb)

        # K graph message-passing rounds over the channel axis.
        # A_hat @ X: for each (B, T_lat) position, mix channels via the
        # normalized adjacency. Reshape so the C axis is the contracted dim.
        for layer in self.msg_layers:
            # x: (B, C, T_lat, H) -> (B, T_lat, H, C) so A_hat (C,C) multiplies.
            x_perm = x.permute(0, 2, 3, 1)          # (B, T_lat, H, C)
            agg = x_perm @ self.a_hat.t()            # (B, T_lat, H, C)
            agg = agg.permute(0, 3, 1, 2)           # (B, C, T_lat, H)
            x = x + torch.nn.functional.gelu(layer(agg))
        return x


class GraphOutProj(nn.Module):
    """Geometry-guided output projection — R2 direction (task 06-29).

    Replaces the dense ``out_proj: Linear(hidden -> C*D)`` (the R1-confirmed
    source of the off-diagonal sparsity artifact). Each channel's velocity is
    produced by mixing its OWN hidden with its graph NEIGHBOURS' hidden (via the
    fixed electrode adjacency), then a per-channel Linear(hidden -> D). This
    breaks the pure per-channel diagonal dominance (channel a's velocity depends
    only on channel a's hidden) that suppressed off-diagonal structure in the
    first R2 attempt, while keeping the mixing geometry-structured (nearby
    electrodes contribute, distant ones do not).

    The neighbour mixing is a FIXED graph propagation (A_hat, not learned), so
    the only learned part is the per-channel Linear. The asymmetry / structure
    of the resulting connectivity comes from the graph_mixer upstream; this layer
    ensures that structure is not collapsed back to a pure diagonal at the exit.

    A ``mix_alpha`` gate controls how much neighbour influence to add: 0.0 =
    pure per-channel (the first-attempt behaviour, diagonal-dominant), 1.0 =
    replace self with neighbour-aggregate. The neighbour term is residual-added
    so the model can learn to scale it via the downstream Linear weights.
    """

    def __init__(self, n_channels: int, latent_dim: int, hidden_ch: int,
                 a_hat: torch.Tensor = None, mix_alpha: float = 0.5):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.mix_alpha = float(mix_alpha)
        # One shared-weight per-channel Linear applied across C.
        self.proj = nn.Linear(hidden_ch, latent_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if a_hat is not None:
            self.register_buffer("a_hat", a_hat.float(), persistent=True)
        else:
            self.register_buffer("a_hat", torch.eye(n_channels), persistent=True)

    def set_adjacency(self, a_hat: torch.Tensor):
        self.a_hat = a_hat.float().to(self.a_hat.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T_lat, hidden) -> (B, C, T_lat, D).

        Mix each channel's hidden with its graph neighbours' hidden before the
        per-channel projection, so the velocity of channel a depends on a
        neighbourhood of channels — not just a itself.
        """
        # Neighbour aggregate: (B, C, T_lat, H) -> A_hat over C axis.
        # x.permute to (B, T_lat, H, C); A_hat (C,C); back to (B, C, T_lat, H).
        x_perm = x.permute(0, 2, 3, 1)             # (B, T_lat, H, C)
        agg = x_perm @ self.a_hat.t()              # (B, T_lat, H, C)
        agg = agg.permute(0, 3, 1, 2)             # (B, C, T_lat, H)
        # Residual blend: self stays, neighbours added with mix_alpha.
        x_mix = x + self.mix_alpha * (agg - x)
        return self.proj(x_mix)


class LearnableMexHatMixer(nn.Module):
    """Mexican-hat band-pass mixer with a learnable residual — R2 B.5 (task 06-29).

    Replaces the Gaussian graph mixer (which low-pass-averaged channel structure)
    with a band-pass operator: a Mexican-hat (Ricker) kernel gives center
    excitation + surround inhibition (lateral inhibition), which preserves
    high-frequency channel structure instead of isotropizing it (verified:
    row-sum CV 63% vs Gaussian's 5.6%). On top of this fixed band-pass prior,
    a learnable residual matrix W lets the model pick up connections the pure
    distance kernel cannot express (long-range functional links).

    Effective adjacency per message-passing round:
        A_eff = mex_hat(d; sigma) + gamma * W       (directed: W not symmetrized)
        X <- A_eff @ X   (over the channel axis C; A[i,j] = channel j drives i)

    sigma is learnable (the center-surround scale adapts to data); W is a full
    C x C residual (zero-init by default, or SC-initialized via ``w_init``). W is
    NOT symmetrized, so A_eff can be directed (task 07-02-sc-init-w). An
    anti-uniformity regularizer is exposed via ``uniformity_penalty`` so the
    trainer can penalize A_eff row-sums becoming too uniform (the
    isotropization failure mode).

    Data flow (C preserved, T_lat preserved):
        [z_t, z_obs] (B, C, T_lat, 2D)
          -> per-channel Linear(2D -> hidden) + time-embed projection
          -> K x { A_eff @ X ; per-channel Linear(hidden->hidden) + GELU }
          -> (B, C, T_lat, hidden)
    """

    def __init__(self, n_channels: int, latent_dim: int, hidden_ch: int,
                 coords: torch.Tensor = None, k_layers: int = 2,
                 gamma: float = 0.1, w_init: torch.Tensor = None,
                 gamma_m: float = 1.0, kernel: str = "ricker",
                 dog_init: dict = None):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.k_layers = k_layers
        self.gamma = float(gamma)
        # kernel selects the distance-kernel shape: "ricker" (1-param Mexican
        # hat, default) or "dog" (4-param Difference-of-Gaussians, task
        # 07-30). DoG decouples excitation/inhibition widths so sigma no longer
        # single-handedly controls the excitatory/inhibitory balance.
        assert kernel in ("ricker", "dog"), f"unknown kernel: {kernel!r}"
        self.kernel = kernel
        # gamma (W weight) and gamma_m (mexhat kernel scale) are stored as
        # buffers so they are checkpointed and recoverable by downstream
        # loaders (needed to reconstruct A_eff from a saved model).
        self.register_buffer("gamma_buf", torch.tensor(float(gamma)), persistent=True)
        # gamma_m scales the (symmetric) distance kernel M. Default 1.0 =
        # unchanged baseline; set 0 to ablate the band-pass prior (task 07-15).
        self.register_buffer("gamma_m", torch.tensor(float(gamma_m)), persistent=True)

        # Per-channel input projections (NO cross-channel mixing here).
        self.in_proj = nn.Linear(latent_dim * 2, hidden_ch)
        # Time embedding projection to hidden (broadcast condition).
        self.t_proj = nn.Linear(latent_dim, hidden_ch)

        # K message-passing per-channel transforms.
        self.msg_layers = nn.ModuleList(
            [nn.Linear(hidden_ch, hidden_ch) for _ in range(k_layers)])

        # Fixed pairwise distance matrix (registered as buffer so it moves
        # with the module and is checkpointed). Shared by both kernels.
        if coords is not None:
            d = torch.cdist(coords.float(), coords.float())
            median_d = float(d[d > 0].median().item())
        else:
            d = torch.zeros(n_channels, n_channels)
            median_d = 0.13
        self.register_buffer("pairwise_d", d.float(), persistent=True)

        # --- distance-kernel parameters (per-kernel) ---
        if kernel == "ricker":
            # Ricker: 1 learnable param sigma (zero-cross pinned at d=sigma).
            self.log_sigma = nn.Parameter(torch.tensor(float(median_d)).log())
        else:
            # DoG: M = Ae*exp(-d^2/2 sigma_e^2) - Ai*exp(-d^2/2 sigma_i^2),
            # with sigma_i>sigma_e and Ae>Ai enforced by reparametrization:
            #   sigma_e = exp(log_sigma_e)
            #   sigma_i = sigma_e + exp(log_delta_sigma)         (> sigma_e)
            #   Ae      = exp(log_Ae)
            #   Ai      = Ae - softplus(log_delta_a)             (< Ae)
            # Constraints hold identically; no clamp / penalty needed.
            dog_init = dog_init or {}
            sigma_e = float(dog_init.get("sigma_e", median_d / 2.0))
            sigma_i = float(dog_init.get("sigma_i", median_d))
            Ae = float(dog_init.get("Ae", 1.0))
            Ai = float(dog_init.get("Ai", 0.6))
            assert sigma_i > sigma_e > 0, (f"DoG init requires sigma_i({sigma_i}) "
                                           f"> sigma_e({sigma_e}) > 0")
            assert Ae > Ai, f"DoG init requires Ae({Ae}) > Ai({Ai})"
            self.log_sigma_e = nn.Parameter(torch.tensor(float(sigma_e)).log())
            self.log_delta_sigma = nn.Parameter(
                torch.tensor(float(sigma_i - sigma_e)).log())
            self.log_Ae = nn.Parameter(torch.tensor(float(Ae)).log())
            # solve softplus(x) = Ae - Ai  ->  x = log(exp(Ae-Ai) - 1)
            delta_a = float(Ae - Ai)
            self.log_delta_a = nn.Parameter(
                torch.tensor(float(math.log(math.expm1(delta_a)))))

        # Learnable residual W (C x C). Zero-init by default (training starts
        # from the pure band-pass Mexican-hat prior, then learns deviations).
        # If w_init is provided, training starts from a given topology instead.
        # W is NOT symmetrized in effective_adjacency(), so it can be directed
        # regardless of init.
        if w_init is not None:
            assert w_init.shape == (n_channels, n_channels), (
                f"w_init shape {tuple(w_init.shape)} != ({n_channels},{n_channels})")
            self.W = nn.Parameter(w_init.clone().float())
        else:
            self.W = nn.Parameter(torch.zeros(n_channels, n_channels))

    def effective_adjacency(self) -> torch.Tensor:
        """Current A_eff = gamma_m * M(d) + gamma * W.

        (C, C), directed. M is the distance kernel selected by ``self.kernel``:
        - "ricker": 1-param Mexican hat (symmetric), zero-cross pinned at d=sigma.
        - "dog": 4-param Difference-of-Gaussians Ae*Ge - Ai*Gi (symmetric),
          decoupling excitation width (sigma_e) from inhibition width (sigma_i).
        M is scaled by gamma_m (default 1.0 = baseline; set 0 to ablate the
        band-pass prior, task 07-15). W is the learnable residual, NOT
        symmetrized — so A_eff can be asymmetric (A[i,j] = channel j drives i).
        """
        M = self._distance_kernel()
        # M is symmetric (pairwise distance); W is directed (not symmetrized).
        # gamma_buf and gamma_m are buffers (recoverable from checkpoint).
        A = self.gamma_m * M + self.gamma_buf * self.W
        return A

    def _distance_kernel(self) -> torch.Tensor:
        """Symmetric distance kernel M(d) over the channel axis, shape (C, C).

        Dispatches on ``self.kernel``:
        - "ricker": ``(1 - d^2/sigma^2) * exp(-d^2/(2 sigma^2))``.
        - "dog":    ``Ae*exp(-d^2/(2 sigma_e^2)) - Ai*exp(-d^2/(2 sigma_i^2))``,
          with sigma_i>sigma_e and Ae>Ai enforced identically by the
          reparametrization in __init__ (no clamp / penalty).
        """
        d2 = self.pairwise_d ** 2
        if self.kernel == "ricker":
            sigma = self.log_sigma.exp()
            s2 = sigma ** 2
            return (1.0 - d2 / s2) * torch.exp(-d2 / (2.0 * s2))
        # DoG
        sigma_e = self.log_sigma_e.exp()
        sigma_i = sigma_e + self.log_delta_sigma.exp()      # > sigma_e, always
        Ae = self.log_Ae.exp()
        Ai = Ae - F.softplus(self.log_delta_a)              # < Ae, always
        return (Ae * torch.exp(-d2 / (2.0 * sigma_e ** 2))
                - Ai * torch.exp(-d2 / (2.0 * sigma_i ** 2)))

    def dog_params(self) -> dict:
        """Reconstructed DoG parameters {sigma_e, sigma_i, Ae, Ai} (dog kernel).

        For introspection / logging / variant tags. Returns {} for ricker.
        """
        if self.kernel != "dog":
            return {}
        sigma_e = float(self.log_sigma_e.detach().exp())
        sigma_i = sigma_e + float(self.log_delta_sigma.detach().exp())
        Ae = float(self.log_Ae.detach().exp())
        Ai = Ae - float(F.softplus(self.log_delta_a.detach()))
        return {"sigma_e": sigma_e, "sigma_i": sigma_i, "Ae": Ae, "Ai": Ai}

    def uniformity_penalty(self) -> torch.Tensor:
        """Anti-uniformity regularizer: penalize small row-sum variance.

        When row-sums of A_eff are nearly constant (var -> 0), the operator
        averages channels uniformly (the isotropization failure mode). This
        returns 1 / (1 + rowsum.var()) so it is large when uniform, small when
        row-sums are spread (structure preserved). The trainer adds
        lambda * this to the loss.
        """
        A = self.effective_adjacency()
        rowsum = A.sum(dim=1)
        # var over channels; +eps for stability.
        var = rowsum.var(unbiased=False) + 1e-6
        return 1.0 / (1.0 + var)

    def sparsity_penalty(self) -> torch.Tensor:
        """L1 sparsity regularizer on A_eff (off-diagonal only).

        Returns mean(|A_eff_ij|) for i != j. Encourages sparse connectivity
        by driving weak edges to exactly zero.
        """
        A = self.effective_adjacency()
        mask = ~torch.eye(self.n_channels, dtype=torch.bool, device=A.device)
        return A.abs()[mask].mean()

    def forward(self, z_t: torch.Tensor, z_obs: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        """z_t, z_obs: (B, C, T_lat, D); t_emb: (B, C, T_lat, D).
        Returns (B, C, T_lat, hidden_ch)."""
        x = torch.cat([z_t, z_obs], dim=-1)        # (B, C, T_lat, 2D)
        x = self.in_proj(x) + self.t_proj(t_emb)    # (B, C, T_lat, hidden)
        A = self.effective_adjacency()              # (C, C)
        for layer in self.msg_layers:
            # A @ X over the channel axis.
            x_perm = x.permute(0, 2, 3, 1)          # (B, T_lat, H, C)
            agg = x_perm @ A.t()                    # (B, T_lat, H, C)
            agg = agg.permute(0, 3, 1, 2)          # (B, C, T_lat, H)
            x = x + torch.nn.functional.gelu(layer(agg))
        return x


class FlowVelocityNet(nn.Module):
    """v(z_t, t, z_obs) for the obs->fc flow, built from dilated 1D convs.

    Operates on (B, C, T_lat, D): the T_lat axis is the temporal sequence the
    dilated convs stride over. Channel interaction is the connectivity-relevant
    site.

    Two modes (task 06-29 R2):
      - graph_mode=False (default, legacy): dense input_proj Linear(C*D*3->hidden)
        flattens and isotropically mixes all channels; per-block 1x1 convs mix
        the already-flattened hidden. Old stage-2 checkpoints load in this mode.
      - graph_mode=True (R2 beta): GraphChannelMixer replaces input_proj (C
        preserved, cross-channel mixing only via K graph-message-passing rounds
        over the electrode adjacency), GraphOutProj replaces out_proj (per-channel
        Linear(hidden->D)), and the dilated convs run with C folded into batch
        (no unstructured cross-channel mixing). All connectivity-relevant mixing
        is concentrated in the graph mixer.

    Dilation schedule {1,2,4,8,16,32} gives a receptive field >> T_lat=16, so
    every output step sees the whole latent sequence.
    """

    def __init__(self, n_channels: int = 128, latent_dim: int = 64,
                 t_lat: int = 16, hidden_ch: int = 128,
                 dilations=(1, 2, 4, 8, 16, 32),
                 graph_mode: bool = False, graph_k: int = 2,
                 a_hat: torch.Tensor = None,
                 graph_out_mix_alpha: float = 0.5,
                 graph_kernel: str = "gaussian",
                 coords: torch.Tensor = None,
                 mexhat_gamma: float = 0.1,
                 w_init: torch.Tensor = None,
                 mexhat_gamma_m: float = 1.0,
                 dog_init: dict = None):
        super().__init__()
        self.t_lat = t_lat
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.graph_mode = graph_mode
        self.graph_kernel = graph_kernel

        self.time_embed = SinusoidalTimeEmbedding(latent_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Dilated 1D conv stack (shared by both modes). In graph mode the C axis
        # is folded into the batch dim, so these convs are strictly per-channel
        # temporal and add no cross-channel mixing.
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.ModuleList([
                nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3,
                          padding=d, dilation=d),
                nn.GroupNorm(8 if hidden_ch % 8 == 0 else 1, hidden_ch),
                nn.GELU(),
                nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),  # channel mix (legacy only)
                nn.GroupNorm(8 if hidden_ch % 8 == 0 else 1, hidden_ch),
                nn.GELU(),
            ]))

        self.out_norm = nn.GroupNorm(8 if hidden_ch % 8 == 0 else 1, hidden_ch)

        if graph_mode:
            if graph_kernel in ("mexhat", "dog"):
                # Band-pass distance-kernel mixer + learnable residual W.
                # "mexhat": 1-param Ricker (default); "dog": 4-param DoG that
                # decouples excitation/inhibition widths (task 07-30). Both
                # share the gamma_m/gamma_buf/W plumbing and the pure
                # per-channel out projection.
                self.graph_mixer = LearnableMexHatMixer(
                    n_channels, latent_dim, hidden_ch, coords=coords,
                    k_layers=graph_k, gamma=mexhat_gamma, w_init=w_init,
                    gamma_m=mexhat_gamma_m,
                    kernel=("ricker" if graph_kernel == "mexhat" else "dog"),
                    dog_init=dog_init)
                # Pure per-channel out (no neighbour mix) so the band-pass
                # structure from the mixer is not re-averaged at the exit.
                self.graph_out = nn.Linear(hidden_ch, latent_dim)
                nn.init.zeros_(self.graph_out.weight)
                nn.init.zeros_(self.graph_out.bias)
            else:
                # R2 B.1-B.4: Gaussian graph mixer + graph-guided out projection.
                self.graph_mixer = GraphChannelMixer(
                    n_channels, latent_dim, hidden_ch, k_layers=graph_k,
                    a_hat=a_hat)
                self.graph_out = GraphOutProj(n_channels, latent_dim, hidden_ch,
                                              a_hat=a_hat, mix_alpha=graph_out_mix_alpha)
        else:
            # Legacy dense path (old checkpoints load here).
            in_ch = n_channels * latent_dim * 3
            self.input_proj = nn.Linear(in_ch, hidden_ch)
            self.out_proj = nn.Linear(hidden_ch, n_channels * latent_dim)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                z_obs: torch.Tensor) -> torch.Tensor:
        # z_t, z_obs: (B, C, T_lat, D); t: (B,)
        B, C, T_lat, D = z_t.shape

        if self.graph_mode:
            return self._forward_graph(z_t, t, z_obs, B, C, T_lat, D)
        return self._forward_dense(z_t, t, z_obs, B, C, T_lat, D)

    def _forward_dense(self, z_t, t, z_obs, B, C, T_lat, D):
        """Legacy dense path (graph_mode=False)."""
        t_emb = self.time_proj(self.time_embed(t))  # (B, D)
        t_emb = t_emb.view(B, 1, 1, D).expand(B, C, T_lat, D)

        x = torch.cat([z_t, z_obs, t_emb], dim=-1)   # (B, C, T_lat, 3D)
        x = x.reshape(B, T_lat, C * D * 3)
        x = self.input_proj(x)                        # (B, T_lat, hidden_ch)
        x = x.permute(0, 2, 1)                        # (B, hidden_ch, T_lat)

        for conv, norm1, act, mix, norm2, _ in [(b[0], b[1], b[2], b[3], b[4], b[5]) for b in self.blocks]:
            residual = x
            x = conv(x)
            x = act(norm1(x))
            x = act(norm2(mix(x)))
            x = x + residual                          # residual over the block

        x = self.out_norm(x)
        x = x.permute(0, 2, 1)                        # (B, T_lat, hidden_ch)
        x = self.out_proj(x)                          # (B, T_lat, C*D)
        x = x.view(B, T_lat, C, D).permute(0, 2, 1, 3)  # (B, C, T_lat, D)
        return x

    def _forward_graph(self, z_t, t, z_obs, B, C, T_lat, D):
        """R2 graph path (graph_mode=True).

        Per-channel mixing is geometry-conditioned: C is preserved through the
        graph mixer, the dilated convs run with C folded into batch (strictly
        per-channel temporal), and the exit is a per-channel projection.
        """
        # Time embedding broadcast to (B, C, T_lat, D).
        t_emb = self.time_proj(self.time_embed(t))   # (B, D)
        t_emb = t_emb.view(B, 1, 1, D).expand(B, C, T_lat, D)

        # Graph mixer: per-channel proj + K message-passing rounds. Output
        # (B, C, T_lat, hidden).
        h = self.graph_mixer(z_t, z_obs, t_emb)      # (B, C, T_lat, hidden)

        # Dilated convs: fold C into batch so they are per-channel temporal.
        # (B, C, T_lat, H) -> (B*C, H, T_lat)
        x = h.permute(0, 1, 3, 2).reshape(B * C, self.hidden_ch, T_lat)
        for conv, norm1, act, mix, norm2, _ in [(b[0], b[1], b[2], b[3], b[4], b[5]) for b in self.blocks]:
            residual = x
            x = conv(x)
            x = act(norm1(x))
            # NOTE: the 1x1 'mix' conv here mixes the hidden FEATURE axis, not
            # channels (C is folded away). Keep it for capacity parity with the
            # legacy path; it adds no cross-channel mixing.
            x = act(norm2(mix(x)))
            x = x + residual

        x = self.out_norm(x)                          # (B*C, hidden, T_lat)
        # (B*C, hidden, T_lat) -> (B, C, T_lat, hidden)
        x = x.permute(0, 2, 1).reshape(B, C, T_lat, self.hidden_ch)
        # Per-channel output projection -> (B, C, T_lat, D)
        x = self.graph_out(x)
        return x


# =============================================================================
# Conv decoder — temporal latent -> full window
# =============================================================================

class ConvDecoder(nn.Module):
    """Per-channel 1D conv decoder with U-Net skip connections.

    Input  : z_obs (B, C, T_lat, D), z_fc (B, C, T_lat, D), skips (optional)
    Output : x_hat (B, T_full, C)   reconstructs obs + fc

    Concatenates z_obs and z_fc on the T_lat axis (so the decoder sees both),
    then upsamples T: 2*T_lat -> ... -> T_full. At each upsampling level, the
    matching encoder skip is concatenated (U-Net style) so high-frequency
    detail bypasses the latent bottleneck — this is what the no-skip decoder
    could not recover (ablation showed ideal-latent Pearson only 0.086).

    skips: list of (B*C, ch_i, T_down_i) from ConvEncoder(return_skips=True),
    ordered shallow->deep. Decoder consumes them deep->shallow.
    """

    def __init__(self, n_channels: int = 128, latent_dim: int = 64,
                 t_lat: int = 16, obs_len: int = 375, forecast_len: int = 125,
                 base_ch: int = 32, n_levels: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.t_lat = t_lat
        self.t_full = obs_len + forecast_len
        self.n_levels = n_levels

        # Per-channel decode: (B*C, 2*t_lat, D) -> ... -> (B*C, 1, T_full)
        # Concat z_obs + z_fc on T_lat axis first.
        # Each level: ConvTranspose upsample, then concat skip (if given),
        # then a 1x1 conv to fuse and taper the channel width.
        self.up_blocks = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()  # 1x1 conv to fuse concat(skip, upsampled)
        in_ch = latent_dim
        for i in range(n_levels):
            out_ch = base_ch * (2 ** (n_levels - 1 - i))  # taper down
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
                nn.GELU(),
            ))
            # skip channels: encoder level (n_levels-1-i) has width
            # base_ch * 2^(n_levels-1-i) == out_ch; fuse = out_ch (upsampled) + out_ch (skip)
            self.skip_fuse.append(nn.Sequential(
                nn.Conv1d(out_ch * 2, out_ch, kernel_size=1),
                nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.flatten_ch = in_ch

        # Final 1x1 conv to map feature width to 1 (per-channel output), then
        # interpolate to exactly T_full (ConvTranspose gives a fixed multiple).
        self.out_conv = nn.Conv1d(self.flatten_ch, 1, kernel_size=1)

    def forward(self, z_obs: torch.Tensor, z_fc: torch.Tensor,
                skips=None) -> torch.Tensor:
        # z_obs, z_fc: (B, C, T_lat, D); skips: list of (B*C, ch_i, T_i) or None
        B, C, T_lat, D = z_obs.shape

        # Concat on T_lat -> (B, C, 2*T_lat, D), then batch channels
        z = torch.cat([z_obs, z_fc], dim=2)           # (B, C, 2*T_lat, D)
        h = z.permute(0, 1, 3, 2).reshape(B * C, D, 2 * T_lat)  # (B*C, D, 2*T_lat)

        for i, block in enumerate(self.up_blocks):
            h = block(h)  # (B*C, out_ch_i, T_up_i)
            if skips is not None:
                # matching encoder skip: level (n_levels-1-i), deep->shallow
                skip = skips[self.n_levels - 1 - i]  # (B*C, ch, T_down)
                # align spatial length via interpolation if needed
                if skip.shape[-1] != h.shape[-1]:
                    skip = F.interpolate(skip, size=h.shape[-1], mode='nearest')
                h = self.skip_fuse[i](torch.cat([h, skip], dim=1))
            # (no skip path: behave like the original decoder)

        # h: (B*C, flatten_ch, T_up) -> (B*C, 1, T_up)
        h = self.out_conv(h)

        # Interpolate to exactly T_full (ConvTranspose multiple may not match)
        if h.shape[-1] != self.t_full:
            h = F.interpolate(h, size=self.t_full, mode='linear', align_corners=False)

        # (B*C, 1, T_full) -> (B, T_full, C)
        h = h.reshape(B, C, self.t_full).permute(0, 2, 1)
        return h


class ConvDecoderFc(nn.Module):
    """Per-channel 1D conv decoder, fc-only (no skip, no z_obs).

    Pure-forecast paradigm (task 08-01): the decoder generates ONLY the fc
    segment (forecast_len samples) from z_fc alone. There is no z_obs
    concatenation and no U-Net skip — z_fc is the sole input, which forces the
    decoder to actually consume it (the diagnosis motivating this was that the
    full-window ConvDecoder structurally drowns z_fc under z_obs + skip: oracle
    feeding encode(x_fc) gave fc Pearson only 0.032, ~equal to feeding a
    non-informative z_obs).

    This differs from the historical no-skip ablation (Pearson 0.086,
    std_ratio 0.30, task 06-23) which still reconstructed the full obs+fc
    window. Here obs reconstruction is dropped entirely, so the decoder has a
    single objective (fc) and a single latent input (z_fc).

    Input  : z_fc (B, C, T_lat, D)
    Output : x_fc_hat (B, forecast_len, C)
    """

    def __init__(self, n_channels: int = 128, latent_dim: int = 64,
                 t_lat: int = 16, forecast_len: int = 125,
                 base_ch: int = 32, n_levels: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.t_lat = t_lat
        self.forecast_len = forecast_len

        # Per-channel decode: (B*C, D, T_lat) -> ... -> (B*C, 1, forecast_len)
        # No skip fusion; input is z_fc only (in_ch = latent_dim, not 2*latent_dim
        # as in ConvDecoder which concatenates z_obs + z_fc on T_lat).
        self.up_blocks = nn.ModuleList()
        in_ch = latent_dim
        for i in range(n_levels):
            out_ch = base_ch * (2 ** (n_levels - 1 - i))  # taper down
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.flatten_ch = in_ch

        # Final 1x1 conv to map feature width to 1 (per-channel output), then
        # interpolate to exactly forecast_len (ConvTranspose gives a fixed multiple).
        self.out_conv = nn.Conv1d(self.flatten_ch, 1, kernel_size=1)

    def forward(self, z_fc: torch.Tensor) -> torch.Tensor:
        # z_fc: (B, C, T_lat, D) -> (B, forecast_len, C)
        B, C, T_lat, D = z_fc.shape

        # Batch channels: (B, C, T_lat, D) -> (B*C, D, T_lat)
        h = z_fc.permute(0, 1, 3, 2).reshape(B * C, D, T_lat)

        for block in self.up_blocks:
            h = block(h)  # (B*C, out_ch_i, T_up_i)

        h = self.out_conv(h)  # (B*C, 1, T_up)

        # Interpolate to exactly forecast_len (ConvTranspose multiple may not match).
        if h.shape[-1] != self.forecast_len:
            h = F.interpolate(h, size=self.forecast_len, mode='linear',
                              align_corners=False)

        # (B*C, 1, forecast_len) -> (B, forecast_len, C)
        h = h.reshape(B, C, self.forecast_len).permute(0, 2, 1)
        return h


# =============================================================================
# Top-level model
# =============================================================================

class OCLatentFlow(nn.Module):
    """Obs-Conditioned Latent Flow model (two-stage).

    Stage 1 (``forward_stage1``): train encoder + proj + decoder with VAE loss.
        The encoder sees only x_obs; z_fc = proj(z_obs); decoder reconstructs
        the full window. KL anneals beta from 0 to beta_final.

    Stage 2 (``forward_stage2``): freeze encoder/proj/decoder, train flow.
        The flow target is the REAL fc latent ``enc(x_fc)`` (same frozen encoder),
        not ``proj(z_obs)``. The flow has NO distribution loss of its own — it
        is optimized only by fc-signal losses (recon/corr/cons) whose gradients
        pass through the frozen decoder into the flow.

    Inference (``sample``): encoder -> flow (Euler, n_steps=10) -> decoder.
    """

    def __init__(self, n_channels: int = 128, latent_dim: int = 64, t_lat: int = 16,
                 obs_len: int = 375, forecast_len: int = 125,
                 enc_base_ch: int = 32, enc_levels: int = 4,
                 dec_base_ch: int = 32, dec_levels: int = 4,
                 flow_hidden_ch: int = 128,
                 flow_graph_mode: bool = False, flow_graph_k: int = 2,
                 flow_a_hat: torch.Tensor = None,
                 flow_graph_out_mix_alpha: float = 0.5,
                 flow_graph_kernel: str = "gaussian",
                 flow_coords: torch.Tensor = None,
                 flow_mexhat_gamma: float = 0.1,
                 flow_w_init: torch.Tensor = None,
                 flow_mexhat_gamma_m: float = 1.0,
                 flow_dog_init: dict = None,
                 # Stage-1 loss weights
                 lambda_td: float = 0.1, lambda_bd: float = 0.05,
                 lambda_corr: float = 1.0, lambda_cons: float = 1.0,
                 beta_kl_final: float = 1e-4,
                 # Stage-2 loss weights
                 lambda_recon2: float = 1.0, lambda_corr2: float = 1.0,
                 lambda_cons2: float = 1.0,
                 # Decoder paradigm: full-window (default) vs fc-only forecast
                 use_fc_decoder: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.t_lat = t_lat
        self.obs_len = obs_len
        self.forecast_len = forecast_len
        self.t_full = obs_len + forecast_len
        self.flow_graph_mode = flow_graph_mode
        self.flow_graph_kernel = flow_graph_kernel

        # Architecture
        self.encoder = ConvEncoder(n_channels, latent_dim, t_lat, enc_base_ch, enc_levels)
        self.proj = ProjHead(latent_dim)
        self.flow = FlowVelocityNet(n_channels, latent_dim, t_lat, flow_hidden_ch,
                                    graph_mode=flow_graph_mode, graph_k=flow_graph_k,
                                    a_hat=flow_a_hat,
                                    graph_out_mix_alpha=flow_graph_out_mix_alpha,
                                    graph_kernel=flow_graph_kernel,
                                    coords=flow_coords,
                                    mexhat_gamma=flow_mexhat_gamma,
                                    w_init=flow_w_init,
                                    mexhat_gamma_m=flow_mexhat_gamma_m,
                                    dog_init=flow_dog_init)
        self.decoder = ConvDecoder(n_channels, latent_dim, t_lat, obs_len, forecast_len,
                                   dec_base_ch, dec_levels)
        # fc-only decoder (pure-forecast paradigm, task 08-01): generates only
        # the fc segment from z_fc alone (no skip, no z_obs). When enabled,
        # forward_stage1/2/sample route to this decoder instead of self.decoder.
        self.use_fc_decoder = use_fc_decoder
        if use_fc_decoder:
            self.decoder_fc = ConvDecoderFc(n_channels, latent_dim, t_lat,
                                            forecast_len, dec_base_ch, dec_levels)
        else:
            self.decoder_fc = None

        # Loss weights
        self.lambda_td = lambda_td
        self.lambda_bd = lambda_bd
        self.lambda_corr = lambda_corr
        self.lambda_cons = lambda_cons
        self.beta_kl_final = beta_kl_final
        self._current_beta = 0.0  # updated externally via set_kl_beta

        self.lambda_recon2 = lambda_recon2
        self.lambda_corr2 = lambda_corr2
        self.lambda_cons2 = lambda_cons2

    # ------------------------------------------------------------------
    # KL annealing hook
    # ------------------------------------------------------------------

    def set_kl_beta(self, beta: float):
        """Called by the training script each epoch for KL annealing (Q10)."""
        self._current_beta = float(beta)

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, x_obs: torch.Tensor, return_skips: bool = False):
        if return_skips:
            mu_obs, logvar_obs, skips = self.encoder(x_obs, return_skips=True)
            return mu_obs, logvar_obs, skips
        mu_obs, logvar_obs = self.encoder(x_obs)
        return mu_obs, logvar_obs

    @staticmethod
    def _reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    # ------------------------------------------------------------------
    # Loss primitives (kept as methods so stage-1 and stage-2 share them)
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_diff(x_hat: torch.Tensor, x_full: torch.Tensor) -> torch.Tensor:
        dx_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]
        dx_true = x_full[:, 1:, :] - x_full[:, :-1, :]
        return (dx_hat - dx_true).abs().mean()

    def _boundary(self, x_hat: torch.Tensor, x_full: torch.Tensor) -> torch.Tensor:
        # obs->fc boundary continuity at sample obs_len-1 -> obs_len
        b_hat = x_hat[:, self.obs_len, :] - x_hat[:, self.obs_len - 1, :]
        b_true = x_full[:, self.obs_len, :] - x_full[:, self.obs_len - 1, :]
        return (b_hat - b_true).abs().mean()

    @staticmethod
    def _pearson_corr_fc(x_hat_fc: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        # per-channel Pearson over time, averaged; returns 1 - corr (lower=better)
        eps = 1e-8
        xc_hat = x_hat_fc - x_hat_fc.mean(dim=1, keepdim=True)
        xc_true = x_fc - x_fc.mean(dim=1, keepdim=True)
        cov = (xc_hat * xc_true).sum(dim=1)
        denom = (xc_hat.norm(dim=1) * xc_true.norm(dim=1) + eps)
        return (1.0 - cov / denom).mean()

    @staticmethod
    def _cons_fc(x_hat_fc: torch.Tensor, x_fc: torch.Tensor) -> torch.Tensor:
        mean_diff = (x_hat_fc.mean(dim=1) - x_fc.mean(dim=1)).abs().mean()
        var_diff = (x_hat_fc.var(dim=1) - x_fc.var(dim=1)).abs().mean()
        return mean_diff + var_diff

    @staticmethod
    def _kl_zobs(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Stage 1: VAE forward (encoder sees only obs)
    # ------------------------------------------------------------------

    def forward_stage1(self, x_obs: torch.Tensor, x_target: torch.Tensor):
        """Stage-1 training forward. Returns (loss, losses_dict).

        x_obs   : (B, obs_len, C)
        x_target: (B, t_full, C)  — full window (obs + fc), ground truth for recon
        """
        B = x_obs.shape[0]
        x_full = x_target[:, :self.t_full, :]
        x_fc = x_full[:, self.obs_len:, :]

        # fc-only paradigm (task 08-01): decoder generates only the fc segment
        # from z_fc alone (no z_obs, no skip). Loss drops obs recon / boundary
        # (no obs output, no obs→fc boundary). z_obs has no reconstruction
        # anchor — KL is the sole posterior constraint (monitor PC1).
        if self.use_fc_decoder:
            mu_obs, logvar_obs = self._encode_obs(x_obs)
            z_obs = self._reparam(mu_obs, logvar_obs)
            z_fc_pred = self.proj(z_obs)
            x_fc_hat = self.decoder_fc(z_fc_pred)          # (B, forecast_len, C)

            recon_fc = (x_fc_hat - x_fc).abs().mean()
            td_fc = self._temporal_diff(x_fc_hat, x_fc)
            corr = self._pearson_corr_fc(x_fc_hat, x_fc)
            cons = self._cons_fc(x_fc_hat, x_fc)
            kl = self._kl_zobs(mu_obs, logvar_obs)

            loss = (
                recon_fc
                + self.lambda_td * td_fc
                + self._current_beta * kl
                + self.lambda_corr * corr
                + self.lambda_cons * cons
            )
            return loss, {
                "recon_fc": recon_fc.detach(),
                "td_fc": td_fc.detach(),
                "corr": corr.detach(),
                "cons": cons.detach(),
                "kl": kl.detach(),
                "beta": torch.tensor(self._current_beta),
            }

        mu_obs, logvar_obs, skips = self._encode_obs(x_obs, return_skips=True)
        z_obs = self._reparam(mu_obs, logvar_obs)      # sampled posterior
        z_fc_pred = self.proj(z_obs)                    # bootstrap fc latent
        x_hat = self.decoder(z_obs, z_fc_pred, skips=skips)  # reconstruct full window

        x_hat_fc = x_hat[:, self.obs_len:, :]

        recon = (x_hat - x_full).abs().mean()
        td = self._temporal_diff(x_hat, x_full)
        bd = self._boundary(x_hat, x_full)
        corr = self._pearson_corr_fc(x_hat_fc, x_fc)
        cons = self._cons_fc(x_hat_fc, x_fc)
        kl = self._kl_zobs(mu_obs, logvar_obs)

        loss = (
            recon
            + self.lambda_td * td
            + self.lambda_bd * bd
            + self._current_beta * kl
            + self.lambda_corr * corr
            + self.lambda_cons * cons
        )

        return loss, {
            "recon": recon.detach(),
            "td": td.detach(),
            "bd": bd.detach(),
            "corr": corr.detach(),
            "cons": cons.detach(),
            "kl": kl.detach(),
            "beta": torch.tensor(self._current_beta),
        }

    # ------------------------------------------------------------------
    # Stage 2: flow forward (encoder/proj/decoder frozen)
    # ------------------------------------------------------------------

    def forward_stage2(self, x_obs: torch.Tensor, x_target: torch.Tensor):
        """Stage-2 training forward. Returns (loss, losses_dict).

        The flow target is the real fc latent ``enc(x_fc)`` from the same
        encoder. The fc-signal losses (recon2/corr2/cons2) drive training; in
        the original frozen-VAE design the flow is optimized only by these
        gradients passing through the decoder.

        A beta-weighted KL term on z_obs is also included. With the frozen-VAE
        design this is a no-op for gradients (logvar is constant), but it
        becomes active under joint training (task 08-01 joint ablation) where
        the encoder is unfrozen — it prevents posterior drift while fc losses
        optimize the encoder/proj/decoder/flow jointly. KL is added only on
        z_obs (the obs posterior), NOT on z_fc_target (the flow target, which
        is a point estimate, not an optimized posterior).
        """
        x_full = x_target[:, :self.t_full, :]
        x_fc = x_full[:, self.obs_len:, :]
        B = x_obs.shape[0]
        device = x_obs.device

        mu_obs, logvar_obs, skips = self._encode_obs(x_obs, return_skips=True)
        z_obs = mu_obs                              # deterministic at stage 2

        # Real fc latent: the encoder is applied to x_fc directly (decision-B
        # fix). Previously the flow target was proj(z_obs), a frozen zero-init
        # linear head — that reduced the flow to copying proj and capped the
        # cross-subject Pearson at proj's own level (~0.20). Encoding the real
        # x_fc gives the flow a genuine latent distribution to transport to,
        # so flow dynamics are learned, not copied from a teacher.
        # The fc encoder uses the SAME frozen encoder weights (no new params,
        # no new trainable surface); it does NOT see obs, so the encoder-still-
        # sees-only-obs constraint for *training* is preserved.
        z_fc_target, _ = self._encode_obs(x_fc)     # (B, C, T_lat, D), real fc latent

        # Flow matching on the z_obs -> z_fc_target line.
        t = torch.rand(B, device=device)
        t_e = t.view(B, 1, 1, 1)
        z_t = (1 - t_e) * z_obs + t_e * z_fc_target
        v_pred = self.flow(z_t, t, z_obs)

        # Single-step fc prediction (latent_flow.py / JET v2 convention).
        # No flow/proj distribution loss: the flow is optimized ONLY by the
        # fc-signal losses below, whose gradients pass through the frozen
        # decoder into the flow. v_pred has no explicit target — its target
        # is whatever latent z_fc_hat makes the decoder's fc output correct.
        z_fc_hat = z_t + (1 - t_e) * v_pred

        # fc-only paradigm (task 08-01): decoder takes z_fc_hat alone (no z_obs,
        # no skip) and outputs fc directly — no obs slice needed.
        if self.use_fc_decoder:
            x_fc_hat_fc = self.decoder_fc(z_fc_hat)
        else:
            x_fc_hat = self.decoder(z_obs, z_fc_hat, skips=skips)
            x_fc_hat_fc = x_fc_hat[:, self.obs_len:, :]

        recon2 = (x_fc_hat_fc - x_fc).abs().mean()
        corr2 = self._pearson_corr_fc(x_fc_hat_fc, x_fc)
        cons2 = self._cons_fc(x_fc_hat_fc, x_fc)
        kl = self._kl_zobs(mu_obs, logvar_obs)

        loss = (
            self.lambda_recon2 * recon2
            + self.lambda_corr2 * corr2
            + self.lambda_cons2 * cons2
            + self._current_beta * kl
        )

        return loss, {
            "recon2": recon2.detach(),
            "corr2": corr2.detach(),
            "cons2": cons2.detach(),
            "kl": kl.detach(),
            "beta": torch.tensor(self._current_beta),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, x_obs: torch.Tensor, n_steps: int = 10,
               use_proj: bool = False) -> torch.Tensor:
        """Inference: encoder -> (flow Euler | proj) -> decoder.

        use_proj=False (default): production path via trained flow (stage 2).
        use_proj=True: stage-1 bootstrap path via proj (for evaluating a VAE
            checkpoint before flow is trained; the untrained flow would corrupt
            the latent, so proj is the correct fc-latent source at stage 1).
        """
        if self.use_fc_decoder:
            mu_obs, _ = self._encode_obs(x_obs)
        else:
            mu_obs, _, skips = self._encode_obs(x_obs, return_skips=True)
        z_obs = mu_obs

        if use_proj:
            z_fc_hat = self.proj(z_obs)
        else:
            # Euler integration of dz/dt = v(z, t, z_obs) from t=0 (z_obs) to t=1
            z = z_obs
            for i in range(n_steps):
                t = torch.full((z.shape[0],), i / n_steps, device=z.device)
                v = self.flow(z, t, z_obs)
                z = z + (1.0 / n_steps) * v
            z_fc_hat = z

        # fc-only paradigm (task 08-01): decoder outputs fc directly.
        if self.use_fc_decoder:
            x_hat = self.decoder_fc(z_fc_hat)
        else:
            x_hat = self.decoder(z_obs, z_fc_hat, skips=skips)
        # Guard against divergence (quality-guidelines: wrap inference in nan_to_num)
        x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)
        return x_hat

    def predict(self, x_obs: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        return self.sample(x_obs, n_steps=n_steps)

    # ------------------------------------------------------------------
    # Self-describing checkpoint seam (task 07-21-arch-checkpoint-seam)
    # ------------------------------------------------------------------
    #
    # An OC-LatentFlow variant is the (topology, knobs) tuple that uniquely
    # determines how to construct the model's FlowVelocityNet. Save writes a
    # small variant header alongside the state-dict (format_version=1); load
    # dispatches on the header. Legacy bare state-dicts (pre-format-v1) are
    # routed through _load_legacy, which sniffs state-dict keys the same way
    # the three previous per-script loaders did.
    #
    # Non-state-dict data that is deterministic from topology (coords via
    # graph_utils.load_geometry(), a_hat via build_normalized_adj()) is NOT
    # stored in the variant dict -- it is recomputed inside _build_from_variant.

    _CHECKPOINT_FORMAT_VERSION = 1

    def _variant_tag(self) -> dict:
        """Read self -> variant dict (inverse of _build_from_variant)."""
        tag = {
            "flow_graph_mode": bool(self.flow_graph_mode),
            "flow_graph_kernel": self.flow_graph_kernel,
            "mexhat_gamma": 0.1,
            "mexhat_gamma_m": 1.0,
            "use_fc_decoder": bool(self.use_fc_decoder),
        }
        if self.flow_graph_mode and self.flow_graph_kernel in ("mexhat", "dog"):
            mixer = getattr(self.flow, "graph_mixer", None)
            if mixer is None:
                return tag
            if hasattr(mixer, "gamma_buf"):
                tag["mexhat_gamma"] = float(mixer.gamma_buf)
            if hasattr(mixer, "gamma_m"):
                tag["mexhat_gamma_m"] = float(mixer.gamma_m)
            # DoG init snapshot (informational; actual params live in state_dict
            # as flow.graph_mixer.log_sigma_e etc.). Captured so a saved DoG
            # checkpoint can be re-constructed with the right kernel path.
            if self.flow_graph_kernel == "dog" and hasattr(mixer, "dog_params"):
                tag["dog_init"] = mixer.dog_params()
        return tag

    @staticmethod
    def _build_from_variant(variant: dict, device) -> "OCLatentFlow":
        """Map variant dict -> constructed OCLatentFlow (no state-dict load).

        Pure: only I/O is recomputing coords/a_hat from the topology choice.
        """
        mode = variant["flow_graph_mode"]
        kernel = variant["flow_graph_kernel"]
        fc_dec = variant.get("use_fc_decoder", False)
        if not mode:
            return OCLatentFlow(use_fc_decoder=fc_dec)  # dense default
        if kernel == "gaussian":
            from latent_flow.models.graph_utils import build_normalized_adj
            a_hat = build_normalized_adj().to(device)
            return OCLatentFlow(flow_graph_mode=True, flow_a_hat=a_hat,
                                use_fc_decoder=fc_dec)
        if kernel == "mexhat":
            from latent_flow.models.graph_utils import load_geometry
            coords, _ = load_geometry()
            return OCLatentFlow(
                flow_graph_mode=True, flow_graph_kernel="mexhat",
                flow_coords=coords,
                flow_mexhat_gamma=variant.get("mexhat_gamma", 0.1),
                flow_mexhat_gamma_m=variant.get("mexhat_gamma_m", 1.0),
                use_fc_decoder=fc_dec,
            )
        if kernel == "dog":
            from latent_flow.models.graph_utils import load_geometry
            coords, _ = load_geometry()
            # dog_init is informational; the learned params come from the
            # state_dict on load. Pass None so the mixer uses median_d-derived
            # defaults, which are immediately overwritten by load_state_dict.
            return OCLatentFlow(
                flow_graph_mode=True, flow_graph_kernel="dog",
                flow_coords=coords,
                flow_mexhat_gamma=variant.get("mexhat_gamma", 0.1),
                flow_mexhat_gamma_m=variant.get("mexhat_gamma_m", 1.0),
                flow_dog_init=variant.get("dog_init"),
                use_fc_decoder=fc_dec,
            )
        raise ValueError(
            f"unknown flow_graph_kernel in variant: {kernel!r}")

    def save_checkpoint(self, path) -> None:
        """Save state_dict + variant header. Self-describing (format v1).

        Existing on-disk checkpoints (bare state-dicts, no header) are still
        loadable via load_checkpoint -> _load_legacy.
        """
        payload = {
            "state_dict": self.state_dict(),
            "variant": self._variant_tag(),
            "format_version": OCLatentFlow._CHECKPOINT_FORMAT_VERSION,
        }
        torch.save(payload, path)

    @staticmethod
    def _load_legacy(state: dict, path, device) -> "OCLatentFlow":
        """Legacy bare state-dict (pre-format-v1). Sniff keys to pick variant.

        Consolidates the three previous per-script sniff paths. Tolerates
        only ``flow.graph_mixer.gamma_m`` as missing (single historical case:
        pre-gamma_m checkpoints predate the buffer). Any other mismatch raises.
        """
        keys = set(state.keys())
        is_mexhat = any(k.startswith("flow.graph_mixer.log_sigma") for k in keys)
        is_graph = any(k.startswith("flow.graph_mixer.") for k in keys)

        if is_mexhat:
            from latent_flow.models.graph_utils import load_geometry
            coords, _ = load_geometry()
            gamma_m = float(state.get("flow.graph_mixer.gamma_m", torch.tensor(1.0)))
            gamma_w = float(state.get("flow.graph_mixer.gamma_buf", torch.tensor(0.1)))
            model = OCLatentFlow(
                flow_graph_mode=True, flow_graph_kernel="mexhat",
                flow_coords=coords,
                flow_mexhat_gamma=gamma_w,
                flow_mexhat_gamma_m=gamma_m)
        elif is_graph:
            from latent_flow.models.graph_utils import build_normalized_adj
            a_hat = build_normalized_adj().to(device)
            model = OCLatentFlow(flow_graph_mode=True, flow_a_hat=a_hat)
        else:
            model = OCLatentFlow()

        missing, unexpected = model.load_state_dict(state, strict=False)
        # gamma_m is a post-hoc buffer added to LearnableMexHatMixer (default
        # 1.0). Pre-addition checkpoints legitimately lack it; the constructor
        # default matches the historical semantics, so tolerating it is safe.
        # sc_base / gamma_sc were removed when the SC prior was deprecated;
        # checkpoints trained before that still carry these keys as unexpected
        # and are silently ignored (the SC contribution is dropped on load).
        allow_missing = {"flow.graph_mixer.gamma_m"}
        allow_unexpected = {"flow.graph_mixer.gamma_sc",
                            "flow.graph_mixer.sc_base"}
        real_missing = set(missing) - allow_missing
        real_unexpected = set(unexpected) - allow_unexpected
        assert not real_missing and not real_unexpected, (
            f"state_dict mismatch loading {path}: "
            f"missing={sorted(real_missing)}, unexpected={sorted(real_unexpected)}")
        return model

    @classmethod
    def load_checkpoint(cls, path, device) -> "OCLatentFlow":
        """Load a checkpoint saved by save_checkpoint or a legacy bare state-dict.

        Dispatches on format_version: v1 -> _build_from_variant + strict load;
        missing header -> _load_legacy sniff path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"checkpoint not found: {path}. "
                "Train OC-LatentFlow stage 2 first "
                "(scripts/train_oc_latent_flow.py --stage 2)."
            )
        payload = torch.load(path, map_location=device, weights_only=False)
        if (isinstance(payload, dict)
                and payload.get("format_version") == cls._CHECKPOINT_FORMAT_VERSION):
            variant = payload["variant"]
            model = cls._build_from_variant(variant, device)
            # strict=False so checkpoints trained before SC deprecation (which
            # carry flow.graph_mixer.sc_base / gamma_sc) load cleanly; the SC
            # contribution is dropped. Any other key mismatch is still a bug.
            model.load_state_dict(payload["state_dict"], strict=False)
        else:
            # Legacy: payload is a bare state-dict (no header).
            model = cls._load_legacy(payload, path, device)
        model.to(device).eval()
        return model
