"""Bisection-based connectivity attribution for forecast models.

Task 06-25-oc-latentflow-connectivity-bisect. Design lives at
``.trellis/tasks/06-25-oc-latentflow-connectivity-bisect/design.md``.

Paradigm
--------
For each source channel ``a`` and target channel ``b``, find the largest
perturbation amplitude ``sigma*^a[b]`` on channel ``a`` such that the
forecast output on channel ``b`` stays within a stability threshold::

    sigma*^a[b] = max sigma in [SIGMA_MIN, SIGMA_MAX]  s.t.  MSE_b(sigma) <= tau_b

where ``MSE_b(sigma)`` is the per-target forecast deviation from the
unperturbed baseline under a Gaussian ``N(0, sigma^2)`` perturbation on
channel ``a``, and ``tau = alpha * median_b(MSE_b(sigma_max))`` is a SINGLE
threshold shared by all targets of this source, equal to a fraction of the
median per-target response at the maximum perturbation.

Why a shared median threshold (not per-target max)
--------------------------------------------------
The earlier per-target form ``tau_b = alpha * MSE_b(sigma_max)`` tied each
target's threshold to its own response. A handful of hyper-responsive
targets (MSE@max up to ~0.18 vs a typical ~8e-5, a 2000x spread) got enormous
thresholds and were clamped to sigma_max — i.e. judged WEAK connections
despite responding strongest. That inverted the connectivity signal and
produced a systematic upper/lower-region split in the heatmap. The median
(robust to the long-tailed response distribution) gives every target one bar;
response strength then flows directly into sigma* (strong response -> small
sigma* -> strong connection). Scale-adaptive per source, but not polluted by
per-target outliers.

Connectivity is then (same formula as ``dropout_entropy.compute_connectivity``,
kept identical so JET/OC results are directly comparable)::

    connectivity[a, b] = -(log sigma*^a[b] + 0.5 * log(2*pi*e))

Large = strong a -> b connection (sigma* small => little perturbation
needed to destabilise b => strong influence).

Why bisection works
-------------------
``MSE_b(sigma)`` is **monotonically non-decreasing** in sigma:
sigma=0 => MSE_b=0 (perturbed == baseline), and larger sigma => larger
input deviation => larger forecast output deviation (the forecast map is
continuous in its input). The set ``{sigma : MSE_b(sigma) <= tau_b}`` is
therefore a contiguous interval ``[0, sigma*]`` and bisection converges to
its upper bound. Strict monotonicity is NOT required (plateaus are fine);
bisection on a monotone set is always correct.

Why bisection (not SGD)
-----------------------
``dropout_entropy.train_sigma_for_source`` trains a learnable sigma via SGD
with backprop through the forecast model — many gradient forwards per source.
Bisection is **forward-only** (no gradients, no ``_sample_impl`` needed): a
fixed sigma grid sweep evaluates the full per-target MSE curve in ``n_grid``
forwards (default 24), then interpolates each target's critical sigma
sub-grid. It also sidesteps the JET stochasticity pitfall (06-19 seeding
lesson): OC-LatentFlow's stage-2 flow path is deterministic
(``z_obs = mu_obs``, no reparameterize), so perturbed-vs-baseline deltas are
clean.

This module does NOT modify ``dropout_entropy.py`` — the SGD/JET path stays
intact as a comparison baseline.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants — kept identical to attribution/dropout_entropy.py so JET (SGD)
# and OC (bisection) connectivity matrices share the same sigma range and
# connectivity formula and are directly comparable.
# ---------------------------------------------------------------------------

SIGMA_MIN: float = 0.01
SIGMA_MAX: float = 0.3
# Half of log(2*pi*e) — the constant part of the gaussian differential
# entropy. Same value as dropout_entropy._HALF_LOG_2PIE.
_HALF_LOG_2PIE: float = 0.5 * math.log(2.0 * math.pi * math.e)

# Bisection convergence: stop when the sigma bracket for a target is narrower
# than this. With range [0.01, 0.3] (width 0.29) this is ~9 bisection rounds.
DEFAULT_PRECISION: float = 1e-3

# Default threshold coefficient for tau = alpha * median_b(MSE_b(sigma_max)).
# alpha=0.5 => the destabilisation bar is HALF the MEDIAN target's full-scale
# response at sigma_max. A target that responds stronger than the median
# clears this bar at a smaller sigma (=> small sigma* => strong connection);
# a weaker-than-median target needs a larger sigma (=> weak connection). This
# lets response strength flow directly into the connectivity signal. The
# earlier per-target tau form (alpha * MSE_b@max) inverted the signal for
# hyper-responsive targets and was replaced by this median-based shared bar.
# alpha=0.5 (not 1.0) was chosen empirically: 1.0 (= the median itself) pushes
# sigma* to sigma_max for nearly all targets (no discrimination); 0.5 places
# the bar in the mid-response range so sigma* spreads out while the earlier
# upper/lower-region split stays fixed (verified: connectivity ratio front/back
# 2.2x -> 0.97).
DEFAULT_ALPHA: float = 0.5

# Zero guard for tau: if a target's max response is ~0 (truly insensitive to
# this source), tau would be 0 and the target clamps to sigma_min by the
# boundary logic. Clamp tau to this floor to avoid divide/log pathologies
# (a near-zero max response already means the target barely responds, which
# the connectivity formula reflects via a small sigma*).
_TAU_EPS: float = 1e-12

DEFAULT_N_STEPS: int = 10          # OC flow Euler steps (quality-guidelines)
DEFAULT_CHUNK_SIZE: int = 16       # target chunk for VRAM control (12 GB)
# Trial batch size per forward. Profiled sweet spot on RTX 5070 Ti Laptop
# (12GB): the ConvDecoder folds the channel axis into its internal batch
# (B*128 rows) and is super-linear there, but the decoder only saturates
# around B=32 (4096 rows). batch=4 (512 rows) underutilises the GPU;
# batch=32 is ~2x faster than batch=4 at n_grid=24 with no accuracy loss
# (sigma* mean stable across batch sizes). batch=64 barely faster but uses
# 4.5GB and the result starts jittering. batch=32 keeps peak VRAM ~2.3GB.
DEFAULT_BATCH_SIZE: int = 32


def default_hparams(**overrides) -> Dict:
    """Default hyperparameters for ``bisect_sigma_for_source``.

    Callers may override any key. The returned dict is what runners echo
    into their output JSON for reproducibility.
    """
    hp = {
        "alpha": DEFAULT_ALPHA,
        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,
        "precision": DEFAULT_PRECISION,
        "n_steps": DEFAULT_N_STEPS,
        "use_proj": False,            # OC-LatentFlow flow path (stage-2)
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "n_grid": 24,                 # sigma grid points (one forward each)
        "amp_dtype": "bfloat16",      # autocast dtype: "bfloat16" | "float16"
        "seed": 42,
    }
    hp.update(overrides)
    return hp


# ---------------------------------------------------------------------------
# Forward / perturbation primitives (forward-only, no grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forecast_fc(model: nn.Module, x_obs: torch.Tensor, n_steps: int,
                use_proj: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    """Run the model forecast and return the fc slice ``x_fc (B, T_fc, C)``.

    Handles both OC-LatentFlow (``sample(x_obs, n_steps, use_proj)`` returns
    the full window; fc = ``[:, obs_len:, :]``) and any model whose
    ``sample`` already returns only the fc segment (defensive: if the output
    time dim equals the configured forecast length, take it as-is). bf16
    autocast for GPU throughput even under no_grad.

    The model is assumed frozen (eval mode, no grad). OC-LatentFlow stage-2
    is deterministic (mu_obs), so repeated calls with the same input give
    identical output — a prerequisite for clean perturbed-vs-baseline deltas.
    """
    device = x_obs.device
    enabled = device.type == "cuda"
    with torch.amp.autocast(
        device_type=("cuda" if device.type == "cuda" else "cpu"),
        dtype=amp_dtype,
        enabled=enabled,
    ):
        out = model.sample(x_obs, n_steps=n_steps, use_proj=use_proj)
    # OC-LatentFlow returns the full obs+fc window (B, T_full, C); fc is the
    # tail. Detect via the model's known lengths when available, else assume
    # the output is already fc-only when its time dim is short.
    obs_len = getattr(model, "obs_len", None)
    if obs_len is not None and out.shape[1] > obs_len:
        out = out[:, obs_len:, :]
    out = torch.nan_to_num(out.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return out


def perturb_channel(x_obs: torch.Tensor, source_channel: int,
                    sigma: float, xi: torch.Tensor,
                    mu_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return a copy of ``x_obs`` with Gaussian noise added to one channel.

    ``x_obs_pert[:, :, source_channel] = x_obs[:, :, source_channel] +
    sigma * xi`` where ``xi ~ N(0, 1)`` of shape ``(B, T_obs)``. Other
    channels are untouched. The caller supplies ``xi`` (drawn from a seeded
    generator) so the SAME noise pattern is used across every sigma in a
    bisection sweep — otherwise MSE(sigma) comparisons would conflate
    amplitude and noise-sample effects.

    When ``mu_bias`` is provided (a ``(C,)`` tensor of per-channel bias
    offsets), an additional fixed-direction bias is added to the perturbed
    channel: ``+ mu_bias[source_channel]``. This enables directional
    (excitation/inhibition) DE connectivity inference. The bias is NOT
    scaled by sigma — it is a fixed direction offset.
    """
    x_pert = x_obs.clone()
    x_pert[:, :, source_channel] = (
        x_pert[:, :, source_channel] + float(sigma) * xi
    )
    if mu_bias is not None:
        x_pert[:, :, source_channel] = (
            x_pert[:, :, source_channel] + mu_bias[source_channel]
        )
    return x_pert


def per_target_mse(x_fc_pert: torch.Tensor,
                   x_fc_base: torch.Tensor) -> torch.Tensor:
    """Per-output-channel MSE between perturbed and baseline fc.

    Args:
        x_fc_pert / x_fc_base: ``(B, T_fc, C)``.

    Returns:
        ``(C,)`` tensor where ``mse[b] = mean over (batch, time) of
        (x_fc_pert[:, :, b] - x_fc_base[:, :, b])**2``.
    """
    sq = (x_fc_pert.float() - x_fc_base.float()) ** 2     # (B, T_fc, C)
    return sq.mean(dim=(0, 1))                              # (C,)


def per_target_mean_diff(x_fc_pert: torch.Tensor,
                         x_fc_base: torch.Tensor) -> torch.Tensor:
    """Per-channel mean shift of perturbed vs baseline fc.

    Used for directional (excitation/inhibition) sign extraction:
    a positive mean shift on target channel ``b`` when perturbing source
    channel ``a`` indicates an excitatory a->b connection; a negative shift
    indicates an inhibitory connection.

    Args:
        x_fc_pert / x_fc_base: ``(B, T_fc, C)``.

    Returns:
        ``(C,)`` tensor where ``diff[b] = mean over (batch, time) of
        (x_fc_pert[:, :, b] - x_fc_base[:, :, b])``.
    """
    diff = x_fc_pert.float() - x_fc_base.float()  # (B, T_fc, C)
    return diff.mean(dim=(0, 1))                   # (C,)


def compute_tau_from_max(mse_at_max: torch.Tensor, alpha: float,
                         eps: float = _TAU_EPS) -> torch.Tensor:
    """Per-target stability threshold ``tau_b = alpha * MSE_b(sigma_max)``.

    The threshold is a fraction of each target's own response at the maximum
    perturbation amplitude (the half-maximum point when ``alpha=0.5``). This
    is scale-invariant across targets and models, and — crucially — it
    always sits INSIDE the response dynamic range, so bisection discriminates
    every target instead of clamping insensitive ones to sigma_max.

    Rationale: an absolute/variance-based threshold was found to be far above
    the forecast model's actual perturbation response (the OC-LatentFlow probe
    showed per-(source,target) MSE responses spanning 3 orders of magnitude,
    max ~0.09, while 0.1*Var(baseline) ~ 0.07 sat at the very top — so 127/128
    targets were "stable at sigma_max"). Anchoring tau to the per-target max
    response makes the critical sigma a normalised "fraction of full
    response" quantity, directly comparable across (source, target) pairs.

    Args:
        mse_at_max: ``(C,)`` per-target MSE at ``sigma = sigma_max``.
        alpha: fraction of the max response (0.5 = half-maximum).
        eps: floor for tau (targets with ~0 max response get a tiny tau, and
            the boundary logic assigns them sigma_min).

    Returns:
        ``(C,)`` tau vector.
    """
    return (alpha * mse_at_max.float()).clamp_min(eps)


def compute_tau_from_median(mse_at_max: torch.Tensor, alpha: float = 1.0,
                            eps: float = _TAU_EPS) -> torch.Tensor:
    """Shared stability threshold ``tau = alpha * median_b(MSE_b(sigma_max))``.

    A SINGLE threshold shared by all targets of this source, equal to a
    fraction of the MEDIAN per-target response at sigma_max. This corrects a
    bias in ``compute_tau_from_max``: that per-target form tied each target's
    threshold to its own response, so a handful of hyper-responsive targets
    (MSE@max up to ~0.18 vs a typical ~8e-5, a 2000x spread) received enormous
    thresholds and were clamped to sigma_max — i.e. judged WEAK connections
    despite responding the strongest. That inverted the connectivity signal
    and produced the upper/lower-region split seen in the heatmap.

    Using the median (robust to the long-tailed response distribution) gives
    every target the same bar: "the sigma at which this target's response
    reaches the typical target's full-scale response". Strong-responding
    targets clear that bar at small sigma (=> small sigma* => strong
    connection), weak-responding targets need large sigma (=> weak
    connection). The signal flows correctly.

    Args:
        mse_at_max: ``(C,)`` per-target MSE at ``sigma = sigma_max``.
        alpha: fraction of the median max response (1.0 = the median itself).
        eps: floor for tau.

    Returns:
        scalar-expanded ``(C,)`` tau vector (same value for every target).
    """
    med = mse_at_max.float().median()
    tau_val = (alpha * med).clamp_min(eps)
    return torch.full_like(mse_at_max.float(), float(tau_val))


# ---------------------------------------------------------------------------
# Connectivity (same formula as dropout_entropy.compute_connectivity)
# ---------------------------------------------------------------------------

def compute_connectivity(sigma_matrix: torch.Tensor) -> torch.Tensor:
    """Connectivity matrix from a sigma matrix.

    ``connectivity[a, b] = -(log sigma*^a[b] + 0.5*log(2*pi*e))``.

    Same formula as ``dropout_entropy.compute_connectivity`` (kept here
    self-contained so this module does not import dropout_entropy — that
    keeps dropout_entropy.py untouched, per the task's zero-regression rule
    for the JET/SGD path). Large = strong a -> b connection.

    Args:
        sigma_matrix: ``(S, C)`` tensor of critical perturbation amplitudes.

    Returns:
        ``(S, C)`` connectivity tensor.
    """
    return -(sigma_matrix.float().log() + _HALF_LOG_2PIE)


# ---------------------------------------------------------------------------
# Core: per-source bisection over ALL targets in parallel
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mse_at_sigma(model: nn.Module, x_obs: torch.Tensor, x_fc_base: torch.Tensor,
                  source_channel: int, sigma: float, xi: torch.Tensor,
                  n_steps: int, use_proj: bool,
                  amp_dtype: torch.dtype) -> torch.Tensor:
    """One perturbed forward at amplitude ``sigma`` -> per-target MSE ``(C,)``.

    ``forecast_fc`` already slices the fc tail internally, so the returned
    tensors are fc-only; the length guard is defensive against an unexpected
    model output shape.
    """
    x_pert = perturb_channel(x_obs, source_channel, sigma, xi)
    x_fc_pert = forecast_fc(model, x_pert, n_steps, use_proj, amp_dtype)
    t = min(x_fc_pert.shape[1], x_fc_base.shape[1])
    return per_target_mse(x_fc_pert[:, :t, :], x_fc_base[:, :t, :])


@torch.no_grad()
def bisect_sigma_for_source(
    model: nn.Module,
    x_obs_subject: torch.Tensor,
    source_channel: int,
    hparams: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Find the critical sigma on every target via a fixed-grid sweep.

    Thin backward-compatible wrapper over :func:`attribution_engine` with a
    :class:`SignalProbeSite`. New callers should construct a ProbeSite and
    call ``attribution_engine`` directly; this entry point is kept so the
    signal-space runner and its tests continue to work unchanged.

    For source channel ``a = source_channel``, finds
    ``sigma*^a[b] = max sigma s.t. MSE_b(sigma) <= tau_b`` for every target
    ``b in [0, C)`` simultaneously, where
    ``tau = alpha * median_b(MSE_b(sigma_max))``.

    Args:
        model: frozen forecast model exposing
            ``sample(x_obs, n_steps, use_proj)`` (OC-LatentFlow stage-2 flow
            path). Must be in eval mode.
        x_obs_subject: ``(N, T_obs, C)`` tensor of one subject's observation
            windows, on ``device``.
        source_channel: index in ``[0, C)`` of the channel to perturb.
        hparams: dict (see :func:`default_hparams`). None => defaults.
        device: torch device. None => ``x_obs_subject.device``.
        verbose: print per-round progress.

    Returns:
        ``(C,)`` float32 tensor of ``sigma*^a[b]`` on CPU. Entries are in
        ``[sigma_min, sigma_max]``.
    """
    hp = default_hparams() if hparams is None else dict(hparams)
    if device is None:
        device = x_obs_subject.device
    if device.type == "cuda":
        amp_name = str(hp.get("amp_dtype", "bfloat16")).lower()
        amp_dtype = torch.float16 if amp_name == "float16" else torch.bfloat16
    else:
        amp_dtype = torch.float32

    site = SignalProbeSite(
        model=model,
        x_obs_subject=x_obs_subject,
        n_steps=int(hp["n_steps"]),
        use_proj=bool(hp["use_proj"]),
        batch_size=int(hp["batch_size"]),
        seed=int(hp["seed"]),
        n_grid=max(int(hp.get("n_grid", 24)), 4),
        sigma_min=float(hp["sigma_min"]),
        sigma_max=float(hp["sigma_max"]),
    )
    site.set_source(source_channel)
    return attribution_engine(
        site=site, source=source_channel,
        alpha=float(hp["alpha"]), device=device,
        amp_dtype=amp_dtype, verbose=verbose,
    )


# ===========================================================================
# ProbeSite — the perturbation-paradigm interface
# ===========================================================================
#
# The signal-space and latent-space attribution runners used to each carry
# their own copy of the bisection algorithm (grid sweep + tau-median + per-
# target monotone walk). They differ ONLY in how a perturbation is applied to
# a source channel and how the per-target response is measured — i.e. the
# "perturb some tensor -> measure some tensor" site. Everything else (the
# sigma search, the threshold, the connectivity formula, the resume/IO loop)
# is identical.
#
# ProbeSite is the seam: a small interface (sigma_grid + perturb_and_measure)
# behind which ALL the site-specific knowledge lives (which tensor is
# perturbed, which axis is the channel, how the forecast runs, how the
# response is reduced, latent-only knobs like condition_mode). The attribution
# engine consumes a ProbeSite and is otherwise site-agnostic, so a new
# perturbation paradigm is a new ProbeSite implementation, not a new runner.
#
# See domain.md "ProbeSite" for the concept.

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class ProbeSite(Protocol):
    """A perturbation paradigm + measurement site for connectivity attribution.

    The attribution engine's only dependency. Implementations encapsulate:
      - which tensor is perturbed and which axis is the channel,
      - how the forecast runs (full encoder->flow->decoder, or flow-only),
      - how the per-target response is measured (which axes to reduce),
      - site-specific knobs (e.g. latent condition_mode, absolute_sigma).

    The baseline (unperturbed response) is the site's private concern — the
    engine never sees it; the site uses it internally inside
    ``perturb_and_measure`` to compute the per-target deviation.
    """

    @property
    def n_targets(self) -> int:
        """Number of target channels (rows in the resulting sigma vector)."""
        ...

    def sigma_grid(self) -> Sequence[float]:
        """The sigma exploration grid for this site (site-chosen resolution).

        Different sites have differently-shaped response curves; the site
        picks the grid density it needs. The engine sweeps this exact
        sequence.
        """
        ...

    def perturb_and_measure(self, sigma: float, source: int) -> torch.Tensor:
        """Apply a ``sigma``-amplitude perturbation to ``source`` -> forecast
        -> measure -> return ``(n_targets,)`` per-target deviation.

        The noise pattern MUST be fixed across calls within one source (drawn
        once from a seeded generator) so the MSE(sigma) curve reflects
        amplitude only, not noise-sample confounds. The baseline
        (unperturbed response) is the site's internal concern.

        The engine calls this under an autocast context; the site should NOT
        manage dtype itself.
        """
        ...


def walk_monotone_curve(mse_curve: torch.Tensor,
                        grid_sigmas: Sequence[float],
                        tau: torch.Tensor,
                        sigma_min: float, sigma_max: float,
                        verbose: bool = False,
                        source_label: Optional[int] = None) -> torch.Tensor:
    """Per-target ``sigma*`` by walking a monotone non-decreasing MSE curve.

    Shared core of the bisection algorithm, factored out so every ProbeSite
    reuses it. ``mse_curve[b]`` is monotone non-decreasing in the grid index,
    so the set ``{sigma : MSE_b(sigma) <= tau_b}`` is a contiguous interval
    ``[0, sigma*]``; we find its upper bound by locating the last stable grid
    point and linearly interpolating the crossing.

    Args:
        mse_curve: ``(n_grid, C)`` per-target MSE at each grid sigma.
        grid_sigmas: length-``n_grid`` sigma values (ascending).
        tau: ``(C,)`` shared-or-per-target threshold.
        sigma_min, sigma_max: clamp range.
        verbose: print the per-source summary line.
        source_label: source index for the verbose line (cosmetic).

    Returns:
        ``(C,)`` float32 tensor of ``sigma*`` on CPU.
    """
    n_grid, C = mse_curve.shape
    grid_list = list(grid_sigmas)
    mse_cpu = mse_curve.detach().cpu()
    tau_cpu = tau.detach().cpu()
    sigma_star = torch.full((C,), sigma_max, dtype=torch.float32)
    n_at_min = 0
    for b in range(C):
        mse_b = mse_cpu[:, b]
        tau_b = tau_cpu[b]
        # Boundary: already destabilised at sigma_min => sigma* = sigma_min.
        if mse_b[0] > tau_b:
            sigma_star[b] = sigma_min
            n_at_min += 1
            continue
        stable = mse_b <= tau_b
        if stable.all().item():
            # Never destabilised within the grid (only possible if alpha>=1).
            sigma_star[b] = sigma_max
            continue
        i = max(int(stable.sum().item()) - 1, 0)
        s_lo, s_hi = grid_list[i], grid_list[i + 1]
        m_lo, m_hi = float(mse_b[i]), float(mse_b[i + 1])
        if m_hi == m_lo:
            sigma_star[b] = s_hi
        else:
            frac = min(max((tau_b - m_lo) / (m_hi - m_lo), 0.0), 1.0)
            sigma_star[b] = s_lo + frac * (s_hi - s_lo)

    if verbose:
        label = f"src {source_label:3d}" if source_label is not None else "src  ?"
        print(
            f"    [{label}] grid={n_grid} points | "
            f"destab@min={n_at_min} | "
            f"sigma* mean={float(sigma_star.mean()):.4f} "
            f"std={float(sigma_star.std()):.4f}"
        )
    return sigma_star.clamp(sigma_min, sigma_max).detach().cpu()


def _extract_mse(result):
    """Unpack the MSE component from a ``perturb_and_measure`` result.

    ``SignalProbeSite.perturb_and_measure`` returns a ``(mse, mean_diff)``
    tuple (task 07-23-directional-noise-sign); ``LatentProbeSite`` returns a
    single ``(C,)`` tensor. This helper normalises both to a single MSE
    tensor so generic callers (``attribution_engine``,
    ``compute_relative_response``) work with either site.
    """
    if isinstance(result, tuple):
        return result[0]
    return result


def attribution_engine(site: "ProbeSite", source: int, alpha: float,
                       device: torch.device,
                       amp_dtype: torch.dtype = torch.bfloat16,
                       verbose: bool = False) -> torch.Tensor:
    """Site-agnostic ``sigma*`` search: grid sweep -> tau-median -> monotone walk.

    This is the single home of the bisection algorithm. It sweeps the site's
    sigma grid (under an autocast context it manages), computes the shared
    median threshold, and walks the resulting monotone per-target curve. The
    site supplies ``perturb_and_measure``; the engine supplies everything
    else. Replacing a perturbation paradigm means a new ProbeSite, not a new
    engine.

    Args:
        site: a ProbeSite implementation (holds the perturbation + forecast +
            measurement knowledge).
        source: source channel index to perturb.
        alpha: threshold coefficient (``tau = alpha * median_b(MSE_b@max)``).
        device: torch device.
        amp_dtype: autocast dtype. bf16 is correct and numerically safe;
            float16 is numerically WRONG for the perturbation-MSE (see
            domain.md 06-25 "fp16 is numerically WRONG"). Default bf16.
        verbose: print the per-source summary.

    Returns:
        ``(n_targets,)`` float32 tensor of ``sigma*`` on CPU.
    """
    grid = site.sigma_grid()
    sigma_min, sigma_max = float(min(grid)), float(max(grid))

    enabled = device.type == "cuda"
    with torch.amp.autocast(
        device_type=("cuda" if device.type == "cuda" else "cpu"),
        dtype=amp_dtype,
        enabled=enabled,
    ):
        mse_curve = torch.stack(
            [_extract_mse(site.perturb_and_measure(float(s), source))
             for s in grid], dim=0
        )  # (n_grid, n_targets)

    tau = compute_tau_from_median(mse_curve[-1], alpha=alpha)
    return walk_monotone_curve(
        mse_curve, grid, tau, sigma_min, sigma_max,
        verbose=verbose, source_label=source,
    )


# ===========================================================================
# Concrete ProbeSites
# ===========================================================================

class SignalProbeSite:
    """Signal-space probe site: perturb ``x_obs[:,:,a]``, measure on ``x_fc``.

    The original OC-LatentFlow connectivity site. Walks the full forecast
    path (encoder -> flow -> decoder) for every perturbed forward. Trials are
    batched over the subject (the ConvDecoder is super-linear in ``B*C``, so a
    tuned ``batch_size`` controls GPU occupancy — see domain.md 06-25).
    """

    def __init__(self, model: nn.Module, x_obs_subject: torch.Tensor,
                 n_steps: int, use_proj: bool, batch_size: int, seed: int,
                 n_grid: int = 24, sigma_min: float = SIGMA_MIN,
                 sigma_max: float = SIGMA_MAX,
                 use_biased_noise: bool = False):
        self.model = model
        self.x_obs = x_obs_subject
        self.n_steps = n_steps
        self.use_proj = use_proj
        self.batch_size = batch_size
        self.seed = seed
        self.n_grid = n_grid
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = x_obs_subject.device
        self.N, _, self.C = x_obs_subject.shape
        self._use_biased_noise = use_biased_noise

        # Per-source noise is drawn lazily (set_source) so the SAME xi is
        # reused across every sigma in a sweep (amplitude-only comparison).
        self._xi = None
        self._source = None
        self._x_fc_base = self._baseline()

        # Per-channel bias for directional perturbation (task
        # 07-23-de-directional-noise-sign). Only active when
        # use_biased_noise=True. mu_bias = 0.5 * std(x_obs_channel)
        # gives a fixed-direction offset that does NOT scale with sigma,
        # enabling excitation/inhibition sign extraction from the mean shift
        # of the FC forecast output.
        if use_biased_noise:
            ch_std = x_obs_subject.float().std(dim=(0, 1))  # (C,)
            self._mu_bias = 0.5 * ch_std
        else:
            self._mu_bias = None

    def _baseline(self) -> torch.Tensor:
        """Unperturbed fc over the whole subject, batched."""
        chunks = []
        for start in range(0, self.N, self.batch_size):
            end = min(start + self.batch_size, self.N)
            chunks.append(
                forecast_fc(self.model, self.x_obs[start:end], self.n_steps,
                            self.use_proj, torch.bfloat16)
            )
        return torch.cat(chunks, dim=0)

    def set_source(self, source: int) -> None:
        """Fix the source channel + draw its seeded noise for the next sweep."""
        self._source = source
        gen = torch.Generator(device=self.device).manual_seed(
            int(self.seed) + source)
        self._xi = torch.randn(self.N, self.x_obs.shape[1],
                               device=self.device, generator=gen)

    @property
    def n_targets(self) -> int:
        return self.C

    def sigma_grid(self) -> Sequence[float]:
        return torch.linspace(self.sigma_min, self.sigma_max, self.n_grid,
                              device=self.device).tolist()

    def perturb_and_measure(self, sigma: float, source: int):
        """Apply perturbation to ``source`` -> forecast -> measure.

        Returns a tuple ``(mse_vec, mean_diff_vec)`` where both are ``(C,)``
        tensors on the device. ``mse_vec[b]`` is the per-target MSE; 
        ``mean_diff_vec[b]`` is the per-target mean shift (positive = 
        excitatory, negative = inhibitory) used for directional sign extraction.

        The mu_bias offset (0.5 * per-channel std, computed at construction) is
        passed to ``perturb_channel`` as a fixed-direction bias that does NOT
        scale with sigma.
        """
        if source != self._source:
            self.set_source(source)
        mse_sum = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        mean_diff_sum = torch.zeros(self.C, device=self.device,
                                    dtype=torch.float32)
        for start in range(0, self.N, self.batch_size):
            end = min(start + self.batch_size, self.N)
            x_pert = perturb_channel(self.x_obs[start:end], source, sigma,
                                     self._xi[start:end],
                                     mu_bias=self._mu_bias)
            x_fc_pert = self.model.sample(x_pert, n_steps=self.n_steps,
                                          use_proj=self.use_proj)
            obs_len = getattr(self.model, "obs_len", None)
            if obs_len is not None and x_fc_pert.shape[1] > obs_len:
                x_fc_pert = x_fc_pert[:, obs_len:, :]
            x_fc_pert = torch.nan_to_num(x_fc_pert.float(), nan=0.0,
                                         posinf=0.0, neginf=0.0)
            t = min(x_fc_pert.shape[1], self._x_fc_base.shape[1])
            mse_sum += per_target_mse(
                x_fc_pert[:, :t, :], self._x_fc_base[start:end][:, :t, :]
            ) * (end - start)
            mean_diff_sum += per_target_mean_diff(
                x_fc_pert[:, :t, :], self._x_fc_base[start:end][:, :t, :]
            ) * (end - start)
        return mse_sum / self.N, mean_diff_sum / self.N


class LatentProbeSite:
    """Latent-space probe site: perturb ``z_obs[:,a,:,:]``, measure on ``z_fc_hat``.

    Bypasses the decoder: the encoder runs ONCE per subject (in the
    constructor), and each perturbed forward integrates only the flow. This
    isolates the flow's own channel mixing from decoder nonlinear
    amplification (see domain.md 06-28 three-layer probe).

    Site-specific knobs (NOT in the ProbeSite interface — construction params):
      - absolute_sigma: if False, sigma is scaled by per-channel latent std
        (relative perturbation); if True, sigma is used directly.
      - condition_mode: "coupled" (cond = perturbed z_obs) or "start_only"
        (cond = clean z_obs). Isolates start-state vs condition-pathway
        selectivity.
      - batch_size: trials per flow-Euler forward (VRAM control). The graph
        mixer (mexhat) runs K rounds of A_eff @ X over all trials kept in
        flight, so the dense-flow assumption that all N trials fit in VRAM
        breaks on 12GB GPUs — see OOM note below. Mirrors SignalProbeSite.
    """

    def __init__(self, model: nn.Module, x_obs_subject: torch.Tensor,
                 n_steps: int, seed: int,
                 absolute_sigma: bool = False,
                 condition_mode: str = "coupled",
                 n_grid: int = 24, sigma_min: float = SIGMA_MIN,
                 sigma_max: float = SIGMA_MAX,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.model = model
        self.n_steps = n_steps
        self.absolute_sigma = absolute_sigma
        self.condition_mode = condition_mode
        self.n_grid = n_grid
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.batch_size = batch_size
        self.device = x_obs_subject.device
        self.seed = seed

        # Encode once for the whole subject; reuse for every source. Detach
        # so the frozen encoder's z_obs carries no grad.
        with torch.no_grad():
            mu_obs, _, _ = model.encoder(x_obs_subject, return_skips=True)
            self.z_obs = mu_obs.detach()                    # (N, C, T_lat, D)
            self.per_ch_std = self.z_obs.std(dim=(0, 2, 3)).clamp_min(1e-8)
        self.N, self.C, self.T_lat, self.D = self.z_obs.shape
        # Baseline fc over the whole subject, BATCHED. The mexhat graph mixer
        # holds all in-flight trials through K rounds of A_eff @ X, so a
        # single whole-subject Euler pass OOMs on a 12GB GPU (RTX 5070 Ti
        # Laptop) once N is ~169. Chunk the trials like SignalProbeSite.
        self._z_fc_base = self._flow_euler_batched(self.z_obs, self.z_obs)
        self._xi = None
        self._source = None

    def _flow_euler_batched(self, z_start: torch.Tensor,
                            z_cond: torch.Tensor) -> torch.Tensor:
        """Batched flow Euler: integrate dz/dt = v(z, t, z_cond) over chunks
        of trials. Mirrors SignalProbeSite's trial batching so the mexhat
        graph mixer's K-round A_eff @ X never holds all N trials at once.
        ``z_start`` and ``z_cond`` are sliced in lockstep; output recombined.
        """
        if z_start.shape[0] <= self.batch_size:
            return self._flow_euler(z_start, z_cond)
        chunks = []
        for s in range(0, z_start.shape[0], self.batch_size):
            e = min(s + self.batch_size, z_start.shape[0])
            chunks.append(self._flow_euler(z_start[s:e], z_cond[s:e]))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _flow_euler(self, z_start: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        """Integrate dz/dt = v(z, t, z_cond) from t=0 to t=1 (Euler, n_steps).

        no_grad is MANDATORY here: perturb_and_measure is pure inference (the
        encoder, decoder and flow are frozen at stage-2). Without it the 24
        grid-point flow forwards inside attribution_engine's autocast (which
        does NOT disable grad) accumulate 24 residual graphs and OOM a 12GB
        GPU at ~28GiB allocated. The signal site avoids this by going through
        ``model.sample`` (which is itself ``@torch.no_grad``); the latent site
        calls ``model.flow`` directly, so the guard lives here.
        """
        z = z_start
        B = z.shape[0]
        for i in range(self.n_steps):
            t = torch.full((B,), i / self.n_steps, device=z.device)
            v = self.model.flow(z, t, z_cond)
            z = z + (1.0 / self.n_steps) * v
        return z

    def set_source(self, source: int) -> None:
        self._source = source
        gen = torch.Generator(device=self.device).manual_seed(
            int(self.seed) + source)
        self._xi = torch.randn(self.N, self.T_lat, self.D,
                               device=self.device, generator=gen)

    @property
    def n_targets(self) -> int:
        return self.C

    def sigma_grid(self) -> Sequence[float]:
        return torch.linspace(self.sigma_min, self.sigma_max, self.n_grid,
                              device=self.device).tolist()

    def perturb_and_measure(self, sigma: float, source: int) -> torch.Tensor:
        if source != self._source:
            self.set_source(source)
        scale = 1.0 if self.absolute_sigma else float(self.per_ch_std[source].item())
        z_pert = self.z_obs.clone()
        z_pert[:, source, :, :] = z_pert[:, source, :, :] + scale * float(sigma) * self._xi
        if self.condition_mode == "coupled":
            z_fc_pert = self._flow_euler_batched(z_pert, z_pert)
        else:  # start_only
            z_fc_pert = self._flow_euler_batched(z_pert, self.z_obs)
        sq = (z_fc_pert.float() - self._z_fc_base.float()) ** 2  # (N, C, T_lat, D)
        return sq.mean(dim=(0, 2, 3))                            # (C,)


# =============================================================================
# ComponentProbeSite — R1 root-cause lock (task 06-29-channel-selectivity)
# =============================================================================
#
# Extends LatentProbeSite with a WEIGHT-SURGERY step: before the subject's
# baseline forward, clone model.flow and mutate one component's trained weights
# so that the clone's cross-channel mixing is structurally altered. The probe
# then measures how that surgical change moves the per-(source,target) channel
# response. Zero retraining — pure inference-time causal test of the trained
# flow's structural tendency.
#
# This isolates WHERE channel selectivity is suppressed in the trained flow:
#   - input_proj_blockdiag : collapse input_proj to per-channel blocks
#                            (each hidden unit sees ONE input channel only)
#                            -> tests whether input_proj's cross-C flatten is
#                               the suppressor (design.md candidate C1)
#   - out_proj_blockdiag   : same idea on out_proj (each output channel sees
#                            hidden units routed from itself) -> candidate C2
#   - hidden_truncate(k)   : zero hidden units [k:] after input_proj and before
#                            the conv stack -> tests hidden-width bottleneck C3
#   - none                 : no surgery (== LatentProbeSite; baseline CV 7.7%)
#
# Flatten order reminder (FlowVelocityNet.forward): input is
# cat([z_t, z_obs, t_emb], dim=-1).reshape(B, T_lat, C*D*3), so the feature
# vector lays out as [ch0(3D), ch1(3D), ...]; channel a occupies columns
# [a*3D : (a+1)*3D] of input_proj.weight.
# =============================================================================


class ComponentProbeSite(LatentProbeSite):
    """Latent-space probe with forward-surgery on one flow component.

    Construction adds ``surgery`` (one of {none, input_proj_blockdiag,
    out_proj_blockdiag, hidden_truncate}) and optional ``surgery_param``
    (the k for hidden_truncate). Everything else (encoder-once, flow Euler,
    perturb_and_measure) is inherited from LatentProbeSite unchanged.

    The trained ``model`` is NEVER mutated: we deepcopy ``model.flow`` and
    install the clone on ``self.model`` (a local attribute), so the caller's
    trained weights are untouched.
    """

    VALID_SURGERIES = (
        "none",
        "input_proj_blockdiag",
        "out_proj_blockdiag",
        "hidden_truncate",
    )

    def __init__(self, model: nn.Module, x_obs_subject: torch.Tensor,
                 n_steps: int, seed: int, surgery: str = "none",
                 surgery_param=None, n_channels: int = 128,
                 latent_dim: int = 64, absolute_sigma: bool = False,
                 condition_mode: str = "coupled", n_grid: int = 24,
                 sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX):
        if surgery not in self.VALID_SURGERIES:
            raise ValueError(
                f"surgery={surgery!r} not in {self.VALID_SURGERIES}")

        self._n_channels = n_channels
        self._latent_dim = latent_dim

        # Clone the flow so the caller's trained weights are never mutated.
        flow_clone = copy.deepcopy(model.flow)
        self._surgery = surgery
        self._surgery_param = surgery_param

        # Apply the surgery to the clone BEFORE LatentProbeSite.__init__ runs
        # the baseline forward (which bakes self._z_fc_base).
        if surgery != "none":
            self._apply_surgery(flow_clone, n_channels, latent_dim,
                                surgery, surgery_param)

        # Swap the model's flow pointer to the surgically-altered clone.
        self._orig_flow_ref = model.flow
        model.flow = flow_clone

        super().__init__(
            model=model, x_obs_subject=x_obs_subject, n_steps=n_steps,
            seed=seed, absolute_sigma=absolute_sigma,
            condition_mode=condition_mode, n_grid=n_grid,
            sigma_min=sigma_min, sigma_max=sigma_max)

    # ------------------------------------------------------------------
    # Surgery implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_surgery(flow: nn.Module, n_channels: int, latent_dim: int,
                       surgery: str, param) -> None:
        """Mutate ``flow``'s weights in place according to ``surgery``."""
        if surgery == "input_proj_blockdiag":
            ComponentProbeSite._blockdiag_input_proj(flow, n_channels, latent_dim)
        elif surgery == "out_proj_blockdiag":
            ComponentProbeSite._blockdiag_out_proj(flow, n_channels, latent_dim)
        elif surgery == "hidden_truncate":
            k = int(param)
            ComponentProbeSite._truncate_hidden(flow, k)
        else:  # "none" — no-op
            pass

    @staticmethod
    def _blockdiag_input_proj(flow: nn.Module, n_channels: int,
                              latent_dim: int) -> None:
        """Collapse input_proj to a block-diagonal map over channels.

        input_proj.weight is (hidden, C*D*3). The input columns group as
        [ch0(3D), ch1(3D), ...]. For each output hidden unit, keep only the
        block of the single input channel with the largest weight norm on that
        unit, zeroing the other C-1 channels' blocks. Result: every hidden unit
        depends on exactly ONE input channel — cross-C mixing at the entry is
        removed, but per-channel projection capacity is preserved.
        """
        w = flow.input_proj.weight.data            # (hidden, C*D*3)
        hidden = w.shape[0]
        block_w = latent_dim * 3                   # features per channel = 3D
        new_w = torch.zeros_like(w)
        for h in range(hidden):
            row = w[h].view(n_channels, block_w)   # (C, 3D)
            block_norms = row.norm(dim=1)          # (C,)
            dom = int(block_norms.argmax().item())
            new_w[h, dom * block_w:(dom + 1) * block_w] = \
                w[h, dom * block_w:(dom + 1) * block_w]
        flow.input_proj.weight.data.copy_(new_w)

    @staticmethod
    def _blockdiag_out_proj(flow: nn.Module, n_channels: int,
                            latent_dim: int) -> None:
        """Collapse out_proj to a block-diagonal map over channels.

        out_proj.weight is (C*D, hidden). Output rows group as
        [ch0(D), ch1(D), ...]. For each input hidden unit (column), keep only
        the block of the single output channel with the largest weight norm on
        that column, zeroing the other C-1 channels' rows. Result: every hidden
        unit can only feed ONE output channel — cross-C reassembly at the exit
        is removed.
        """
        w = flow.out_proj.weight.data             # (C*D, hidden)
        hidden = w.shape[1]
        block_w = latent_dim                       # = D
        new_w = torch.zeros_like(w)
        for h in range(hidden):
            col = w[:, h].view(n_channels, block_w)   # (C, D)
            block_norms = col.norm(dim=1)          # (C,)
            dom = int(block_norms.argmax().item())
            new_w[dom * block_w:(dom + 1) * block_w, h] = \
                w[dom * block_w:(dom + 1) * block_w, h]
        flow.out_proj.weight.data.copy_(new_w)

    @staticmethod
    def _truncate_hidden(flow: nn.Module, k: int) -> None:
        """Zero hidden units [k:] of input_proj output via a forward hook.

        Instead of rewriting every downstream conv's weights, install a hook
        on input_proj that zeroes rows [k:] of its output. This emulates a
        narrower bottleneck (hidden_ch = k) without changing conv shapes.
        The hook is stored on the flow so it persists for the probe's lifetime.
        """
        def _zero_tail(module, args, output):
            # output: (B, T_lat, hidden). Zero hidden indices >= k in place.
            output[..., k:] = 0.0
            return output

        flow._surgery_hook = flow.input_proj.register_forward_hook(_zero_tail)

    @property
    def surgery_desc(self) -> str:
        if self._surgery == "none":
            return "none"
        if self._surgery == "hidden_truncate":
            return f"hidden_truncate(k={self._surgery_param})"
        return self._surgery


# ===========================================================================
# Per-edge relative-response paradigm (task 07-04-redesign-bisection-paradigm)
# ===========================================================================
#
# The original bisection engine (attribution_engine + compute_tau_from_median
# + walk_monotone_curve above) uses a SHARED PER-SOURCE threshold
# ``tau = median_b(MSE_b(sigma_max))``. Variance decomposition of the resulting
# connectivity matrix (task 07-04-de-vs-baseline-zero-corr) showed that 61% of
# the variance comes from the SOURCE main effect (row), only 35% from the
# true a->b residual: sigma* encodes "source a's global destructiveness", not
# "the a->b directed edge". The matrix is also subject-invariant (inter-subject
# Spearman 0.90 vs GC/TE 0.40) and correlates ~0 with GC/TE.
#
# The functions below implement an alternative read-off that directly targets
# the per-edge (a->b) signal by DOUBLE normalisation, eliminating both the
# source main effect (row) and the target main effect (column):
#
#   raw_response[a, b] = MSE_b^a(sigma_ref)            # one forward per source
#   connectivity[a, b] = median_polish(raw_response)[a, b]  # ANOVA residual
#
# Median polish (Tukey) iteratively subtracts row and column medians to fit
# the two-way additive model response = mu + row_a + col_b + residual[a,b],
# returning the residual. This provably drives both main-effect variances
# toward zero while preserving the per-edge residual structure. After polish,
# the residual is z-scored per source row for cross-source comparability.
#
# It needs ONE forward per source at a fixed sigma_ref (not a 24-point grid
# sweep), so it is ~24x cheaper than bisection.
#
# These functions do NOT modify the original bisection path. They reuse
# ProbeSite.perturb_and_measure (the site's perturbation+forecast+measurement
# seam) but bypass the tau/grid/walk machinery entirely.


@torch.no_grad()
def compute_relative_response(site: "ProbeSite", source: int,
                              sigma_ref: float, device: torch.device,
                              amp_dtype: torch.dtype = torch.bfloat16
                              ) -> torch.Tensor:
    """Per-target MSE response of perturbing ``source`` at a fixed ``sigma_ref``.

    This is the raw material for the per-edge relative-response paradigm: for
    each source channel ``a``, one forward at amplitude ``sigma_ref`` yields
    the per-target deviation ``MSE_b^a(sigma_ref)`` for every target ``b``
    simultaneously. No tau threshold, no grid sweep, no bisection.

    Reuses ``site.perturb_and_measure`` (the site's perturb+forecast+measure
    seam) so the new paradigm works with any existing ProbeSite
    (SignalProbeSite, LatentProbeSite) without site-specific code.

    Args:
        site: a ProbeSite implementation (holds perturbation + forecast +
            measurement knowledge, same as for bisection).
        source: source channel index to perturb.
        sigma_ref: the single perturbation amplitude to evaluate. Unlike
            bisection's sigma_max this is NOT a sweep bound; it is the one
            amplitude at which the per-edge response is read off. Chosen to
            sit in the model's informative response range (default 0.3, the
            former sigma_max).
        device: torch device.
        amp_dtype: autocast dtype (bf16 default, matching attribution_engine).

    Returns:
        ``(n_targets,)`` float32 tensor of per-target MSE on CPU. Entry ``b``
        is ``MSE_b^source(sigma_ref)``.
    """
    enabled = device.type == "cuda"
    with torch.amp.autocast(
        device_type=("cuda" if device.type == "cuda" else "cpu"),
        dtype=amp_dtype,
        enabled=enabled,
    ):
        return _extract_mse(
            site.perturb_and_measure(float(sigma_ref), source)
        ).detach().cpu()


def build_connectivity_relative(response_matrix: torch.Tensor,
                                row_norm: bool = True,
                                col_norm: bool = True,
                                n_iter: int = 5) -> torch.Tensor:
    """Per-edge connectivity by double normalisation of the response matrix.

    Eliminates the two main effects that dominate the original bisection
    connectivity (source destructiveness and target sensitivity), leaving the
    a->b-specific residual signal.

    Method: iterative additive (median-polish) removal of row and column
    effects, equivalent to fitting the two-way additive model
    ``response[a,b] = mu + row_a + col_b + residual[a,b]`` and returning the
    residual. A single subtraction pass leaves a residual because row and
    column effects are entangled; a few alternating iterations (Tukey median
    polish, default 5) converge to a stable residual. This is the standard
    technique for two-way ANOVA decomposition and provably drives both the
    row and column main-effect variances toward zero while preserving the
    per-edge residual structure.

    After median polish, the residual is z-scored per source row (divide by
    row std) so the connectivity values are comparable across sources of
    different dynamic range.

    Args:
        response_matrix: ``(S, C)`` tensor where ``[a, b] = MSE_b^a(sigma_ref)``
            collected over all sources (one row per source, from
            :func:`compute_relative_response`).
        row_norm: if True (default), remove the source (row) main effect and
            z-score the residual per row. Set False to ablate.
        col_norm: if True (default), remove the target (column) main effect.
            Set False to ablate.
        n_iter: number of alternating median-polish iterations (default 5;
            convergence is typically reached in 3-5).

    Returns:
        ``(S, C)`` float32 connectivity tensor (the a->b residual). NaN-safe:
        non-finite entries from degenerate rows/columns are zeroed.
    """
    r = response_matrix.float().clone()
    S, C = r.shape

    # Median polish: alternately subtract row medians and column medians.
    # Converges to the two-way additive ANOVA decomposition residual.
    for _ in range(int(n_iter)):
        if row_norm:
            row_med = r.median(dim=1, keepdim=True).values  # (S, 1)
            r = r - row_med
        if col_norm:
            col_med = r.median(dim=0, keepdim=True).values  # (1, C)
            r = r - col_med

    # Optional per-row z-score for cross-source comparability (only after
    # residuals are formed, so it does not reintroduce a row main effect).
    if row_norm:
        row_std = r.std(dim=1, keepdim=True).clamp_min(1e-12)  # (S, 1)
        r = r / row_std

    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r.detach().cpu()


# ===========================================================================
# Global-quantile-tau paradigm (task 07-04-global-tau-sigma)
# ===========================================================================
#
# CORRECTS the median-polish direction (task 07-04-redesign, archived as a
# control experiment). Median polish discarded the sigma* search — the
# physical content of discard-entropy connectivity (sigma*[a,b] = b's
# tolerance to a's perturbation = a->b connection strength). This module
# PRESERVES the full perturb -> measure -> threshold -> sigma* -> connection
# chain, and fixes only the threshold design.
#
# Root cause of the original failure (task 07-04-de-vs-baseline-zero-corr):
# `tau = median_b(MSE_b^a(sigma_max))` is a SHARED PER-SOURCE threshold, so
# sigma* encoded each source's global destructiveness (78% row variance)
# rather than the per-edge a->b tolerance.
#
# Fix: tau = a single quantile of the GLOBAL MSE response distribution over
# all (a,b) pairs. Every source and target shares the SAME threshold, so
# sigma* values are directly comparable across the matrix. The sigma* search
# (find the critical perturbation amplitude at which target b destabilises)
# is preserved — only the threshold is globalised.
#
# Two-stage algorithm (tau depends on the global distribution):
#   Stage 1: sweep each source once at sigma_ref -> (S,C) raw response matrix
#            -> tau = quantile_q(flatten(response_matrix)).
#   Stage 2: for each source, sweep a sigma grid -> per-target MSE curves ->
#            sigma*[a,b] = largest sigma with MSE_b^a(sigma) <= tau (the
#            monotone-walk critical point; clamped to sigma_max if the target
#            never reaches tau within the grid = weakest connection).


def compute_global_tau(response_matrix: torch.Tensor, q: float) -> float:
    """Single global threshold = q-quantile of the MSE response distribution.

    Flattens the (S, C) raw response matrix (one forward per source at
    sigma_ref) and takes the q-quantile. Every (a,b) pair then faces the SAME
    threshold, unlike the original ``compute_tau_from_median`` which gave each
    source its own threshold (the root cause of the source-main-effect
    dominance).

    Args:
        response_matrix: ``(S, C)`` tensor of ``MSE_b^a(sigma_ref)``, e.g. from
            :func:`compute_relative_response` stacked over sources.
        q: quantile in [0, 1]. 0.5 = the global median response.

    Returns:
        Scalar tau (same units as MSE). All (a,b) pairs share this threshold.
    """
    vals = response_matrix.float().flatten()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return float("nan")
    return float(torch.quantile(vals, float(q)))


def log_grid(sigma_min: float, sigma_max: float, n_grid: int) -> list:
    """Log-spaced sigma grid (denser at small sigma where MSE changes fastest).

    The MSE(sigma) response curve is steep at small sigma (slight perturbation
    already moves the forecast) and plateaus at large sigma, so a log grid
    gives finer resolution where sigma* discrimination matters. Replaces the
    original linear ``linspace(sigma_min, sigma_max, n_grid)`` used by
    ``SignalProbeSite.sigma_grid``.

    Args:
        sigma_min, sigma_max: grid bounds (> 0).
        n_grid: number of grid points.

    Returns:
        Ascending list of sigma values, log-spaced.
    """
    import math as _math
    lo = _math.log(float(sigma_min))
    hi = _math.log(float(sigma_max))
    return [float(x) for x in torch.exp(torch.linspace(lo, hi, int(n_grid)))]


def sigma_star_with_global_tau(
    mse_curve: torch.Tensor,
    grid_sigmas: Sequence[float],
    tau_global: float,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Per-target sigma* under a single global threshold.

    Same monotone-curve walk as :func:`walk_monotone_curve`, but ``tau`` is a
    SCALAR shared by all targets (not a per-source median vector). This is the
    core of the global-tau paradigm: every (a,b) pair is judged by the same
    bar, so sigma* values are directly comparable across the matrix.

    Boundary clamping (preserves the discard-entropy semantics):
      - If ``MSE_b(sigma_min) > tau_global``: target already destabilised at
        the smallest perturbation -> sigma* = sigma_min (strongest connection).
      - If ``MSE_b(sigma_max) < tau_global``: target never reaches tau even at
        the largest perturbation -> sigma* = sigma_max (weakest connection;
        a's perturbation barely reaches b).

    Args:
        mse_curve: ``(n_grid, C)`` per-target MSE at each grid sigma.
        grid_sigmas: length-``n_grid`` sigma values (ascending).
        tau_global: scalar global threshold.
        sigma_min, sigma_max: clamp range.

    Returns:
        ``(C,)`` float32 tensor of sigma* on CPU.
    """
    return walk_monotone_curve(
        mse_curve,
        grid_sigmas,
        torch.full((mse_curve.shape[1],), float(tau_global)),
        sigma_min,
        sigma_max,
    )
