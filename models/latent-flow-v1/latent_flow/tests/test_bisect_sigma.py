"""Unit tests for attribution/bisect_sigma.py (task 06-25).

Verifies the bisection connectivity attribution:

1. ``compute_tau`` — per-target threshold, zero-variance clamp.
2. ``compute_connectivity`` — formula matches dropout_entropy (numerically).
3. ``perturb_channel`` — only the source channel changes.
4. ``bisect_sigma_for_source`` — converges to the analytic critical sigma on
   a dummy model with a KNOWN monotonic MSE(sigma) response; boundary cases
   (destabilised at sigma_min => sigma_min; never destabilised => sigma_max).
5. ``per_target_mse`` — correct mean over batch/time.

Run:

    D:\\program\\anaconda\\envs\\npi\\python.exe -m pytest \\
        tests/test_bisect_sigma.py -q

Per logging-guidelines.md, all output is ASCII (Windows GBK safe).
"""

from __future__ import annotations

import math

import pytest
import torch

from latent_flow.attribution.bisect_sigma import (
    _HALF_LOG_2PIE,
    SIGMA_MAX,
    SIGMA_MIN,
    bisect_sigma_for_source,
    compute_connectivity,
    compute_tau_from_max,
    compute_tau_from_median,
    default_hparams,
    per_target_mse,
    perturb_channel,
)


# ---------------------------------------------------------------------------
# Dummy model with a KNOWN, per-target sensitivity to channel-a perturbation.
# ---------------------------------------------------------------------------
#
# The model ignores x_obs content and returns a baseline fc plus a response
# proportional to the perturbation on the source channel. Concretely, for a
# fixed source channel `a`, the dummy measures how much channel `a` was
# perturbed (by comparing to the unperturbed input it stored) and emits
#     x_fc_pert[:,:,b] = x_fc_base[:,:,b] + gain[b] * delta_a
# where delta_a is the per-(batch,time) perturbation magnitude on channel a.
# Then
#     MSE_b(sigma) = gain[b]^2 * mean(delta_a^2) = gain[b]^2 * sigma^2
# (since xi ~ N(0,1), mean(xi^2) -> 1 over enough samples). So the critical
# sigma satisfying MSE_b = tau_b is
#     sigma*_b = sqrt(tau_b) / gain[b]
# which we can compare the bisection output against analytically.
#
# To make the model "see" the perturbation, it stores the baseline x_obs at
# the first forward and recomputes delta on subsequent forwards. This is a
# test-only trick (real models are stateless w.r.t. x_obs); the bisection
# code only calls model.sample, so the statefulness is invisible to it.

class _PerturbationSensitiveModel(torch.nn.Module):
    """Dummy forecast model: baseline + linear response to channel-a noise.

    The model holds the CLEAN (unperturbed) subject x_obs and a fixed baseline
    fc. On each forward it matches the incoming (possibly perturbed, possibly
    batched) x_obs rows to the stored clean subject by row index — the first
    column of channel 0 is used as a row-identity key (the bisection perturbs
    only `source_channel`, which is != 0 in our tests, so channel 0 col 0 is
    a stable key). The perturbation on `source_channel` is then recovered
    exactly and the per-target response is gain[b] * delta, giving a known
    MSE_b(sigma) = gain[b]^2 * sigma^2 for analytic checks.

    Args:
        n_channels, obs_len, fc_len: shapes.
        source_channel: channel whose perturbation each target responds to.
        gain: (C,) tensor; target b response amplitude = gain[b] * delta.
    """

    def __init__(self, n_channels, obs_len, fc_len, source_channel, gain,
                 clean_subject):
        super().__init__()
        self.n_channels = n_channels
        self.obs_len = obs_len
        self.fc_len = fc_len
        self.t_full = obs_len + fc_len
        self.source_channel = source_channel
        self.register_buffer("gain", gain.float())
        # Store the clean subject so we can recover the perturbation per row.
        self.register_buffer("clean_subject", clean_subject.detach().clone())
        g = torch.Generator().manual_seed(0)
        self.register_buffer(
            "base_fc",
            torch.randn(clean_subject.shape[0], fc_len, n_channels,
                        generator=g),
        )
        g2 = torch.Generator().manual_seed(1)
        # Fixed fluctuation shape (unit std) so MSE is proportional to amp^2.
        shape = torch.randn(clean_subject.shape[0], fc_len, n_channels,
                            generator=g2)
        self.register_buffer("shape", shape / shape.std())

    def _match_rows(self, x_obs):
        """Return the indices of x_obs rows within clean_subject, keyed by
        channel-0 column 0 (stable because the perturbation is on
        ``source_channel != 0``)."""
        key = x_obs[:, 0, 0]                              # (B,)
        clean_key = self.clean_subject[:, 0, 0]           # (N,)
        # argmatch: for each key find the clean row with equal key.
        idx = (key[:, None] == clean_key[None, :]).float().argmax(dim=1)
        return idx

    @torch.no_grad()
    def sample(self, x_obs, n_steps=10, use_proj=False):
        B, T, C = x_obs.shape
        idx = self._match_rows(x_obs)                     # (B,)
        clean = self.clean_subject[idx]                    # (B, T, C)
        # Perturbation on the source channel.
        delta = (x_obs - clean)[:, :, self.source_channel]   # (B, T_obs)
        delta_rms = (delta ** 2).mean(dim=1, keepdim=True).sqrt()  # (B, 1)
        amp = delta_rms * self.gain.unsqueeze(0)           # (B, C)
        base_fc_b = self.base_fc[idx]                      # (B, fc, C)
        shape_b = self.shape[idx]                          # (B, fc, C)
        x_fc = base_fc_b + amp.unsqueeze(1) * shape_b     # (B, fc, C)
        return torch.cat([x_obs, x_fc], dim=1)             # full window


# ---------------------------------------------------------------------------
# Test dims (small for fast CPU tests)
# ---------------------------------------------------------------------------

C, OBS_LEN, FC_LEN = 6, 40, 20
SRC = 2


# ---------------------------------------------------------------------------
# 1. compute_tau_from_max
# ---------------------------------------------------------------------------

def test_compute_tau_from_max_zero_response_clamp():
    """A target with ~0 max response gets tau clamped to eps > 0."""
    mse_at_max = torch.tensor([0.0, 0.5, 1e-15])
    tau = compute_tau_from_max(mse_at_max, alpha=0.5)
    assert tau.shape == (3,)
    # Channels 0 and 2 have (near) zero max response -> clamped to eps.
    assert tau[0].item() > 0
    assert tau[2].item() > 0
    # Channel 1 follows alpha * mse_at_max.
    assert math.isclose(tau[1].item(), 0.5 * 0.5, rel_tol=1e-6)


def test_compute_tau_from_max_scales_with_alpha():
    mse_at_max = torch.tensor([0.1, 0.3, 0.9])
    tau1 = compute_tau_from_max(mse_at_max, alpha=0.3)
    tau2 = compute_tau_from_max(mse_at_max, alpha=0.7)
    # tau proportional to alpha for each target.
    for b in range(3):
        assert math.isclose(tau2[b].item() / tau1[b].item(),
                            0.7 / 0.3, rel_tol=1e-4)


def test_compute_tau_from_max_per_target_adaptive():
    """tau tracks each target's own max response (not a global value)."""
    mse_at_max = torch.tensor([0.01, 1.0, 100.0])     # 4 orders of magnitude
    tau = compute_tau_from_max(mse_at_max, alpha=0.5)
    assert torch.allclose(tau, 0.5 * mse_at_max)
    # The 4-order spread (100/0.01 = 10000) is preserved -> bisection
    # discriminates every target.
    assert math.isclose((tau[2] / tau[0]).item(),
                        (mse_at_max[2] / mse_at_max[0]).item(), rel_tol=1e-6)


def test_compute_tau_from_median_is_shared_and_robust():
    """tau is ONE value shared by all targets = alpha * median(MSE@max).

    This is the bias fix: the per-target max form let hyper-responsive
    targets (long tail) inflate their own threshold. The median form ignores
    outliers and gives every target the same bar.
    """
    # Long-tailed: one hyper-responsive target (0.18) among typical ~8e-5.
    mse_at_max = torch.tensor([8e-5, 7e-5, 9e-5, 0.18, 8.5e-5])
    tau = compute_tau_from_median(mse_at_max, alpha=1.0)
    # Every target shares the SAME tau = median (8.5e-5), not 0.18.
    assert torch.allclose(tau, torch.full_like(tau, 8.5e-5))
    # Robust: the outlier 0.18 does NOT raise the threshold (median ~ unaffected).
    assert tau[3].item() < 0.18


def test_compute_tau_from_median_scales_with_alpha():
    mse_at_max = torch.tensor([0.1, 0.3, 0.9, 0.5])   # median 0.4
    tau1 = compute_tau_from_median(mse_at_max, alpha=0.5)
    tau2 = compute_tau_from_median(mse_at_max, alpha=2.0)
    for b in range(4):
        assert math.isclose(tau2[b].item() / tau1[b].item(), 4.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 2. compute_connectivity — formula parity with dropout_entropy
# ---------------------------------------------------------------------------

def test_compute_connectivity_formula():
    """connectivity[a,b] = -(log sigma + 0.5 log(2 pi e))."""
    sigma = torch.tensor([[0.05, 0.1, 0.3]])
    conn = compute_connectivity(sigma)
    expected = -(sigma.log() + _HALF_LOG_2PIE)
    assert torch.allclose(conn, expected)
    # Stronger connection (smaller sigma) => larger connectivity.
    assert conn[0, 0] > conn[0, 1] > conn[0, 2]


# ---------------------------------------------------------------------------
# 3. perturb_channel
# ---------------------------------------------------------------------------

def test_perturb_channel_only_source_changes():
    x = torch.ones(3, 10, C)
    xi = torch.randn(3, 10)
    x_pert = perturb_channel(x, source_channel=SRC, sigma=0.2, xi=xi)
    # Source channel changed.
    assert not torch.allclose(x_pert[:, :, SRC], x[:, :, SRC])
    # All other channels unchanged.
    for b in range(C):
        if b != SRC:
            assert torch.allclose(x_pert[:, :, b], x[:, :, b])
    # Original tensor not mutated.
    assert torch.allclose(x, torch.ones(3, 10, C))


# ---------------------------------------------------------------------------
# 4. per_target_mse
# ---------------------------------------------------------------------------

def test_per_target_mse():
    a = torch.zeros(2, 5, 3)
    a[:, :, 0] = 1.0
    a[:, :, 1] = 2.0
    b = torch.zeros(2, 5, 3)
    mse = per_target_mse(a, b)
    assert mse.shape == (3,)
    assert math.isclose(mse[0].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(mse[1].item(), 4.0, rel_tol=1e-6)
    assert math.isclose(mse[2].item(), 0.0, abs_tol=1e-7)


# ---------------------------------------------------------------------------
# 5. bisect_sigma_for_source — analytic convergence on the dummy model
# ---------------------------------------------------------------------------
#
# With tau_b = alpha * MSE_b(sigma_max) and MSE_b(sigma) = gain[b]^2 * sigma^2,
# the critical sigma satisfies
#     gain^2 * sigma*^2 = alpha * gain^2 * sigma_max^2
#     => sigma* = sigma_max * sqrt(alpha)
# which is INDEPENDENT of gain. This is the clean analytic target: bisection
# should recover sigma_max * sqrt(alpha) for every target within `precision`.


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_bisect_converges_to_analytic_sigma(alpha):
    """sigma* should equal sigma_max * sqrt(alpha), gain-independent."""
    x_obs = torch.randn(32, OBS_LEN, C)
    model = _PerturbationSensitiveModel(
        n_channels=C, obs_len=OBS_LEN, fc_len=FC_LEN,
        source_channel=SRC, gain=torch.ones(C), clean_subject=x_obs,
    )
    target_sigma = SIGMA_MAX * math.sqrt(alpha)

    sigma_star = bisect_sigma_for_source(
        model, x_obs, source_channel=SRC,
        hparams=default_hparams(alpha=alpha, precision=1e-3, batch_size=8),
        device=torch.device("cpu"),
        verbose=False,
    )
    assert sigma_star.shape == (C,)
    err = (sigma_star - target_sigma).abs()
    assert err.max().item() < 2e-2, (
        f"max |sigma* - sigma_max*sqrt({alpha})={target_sigma:.4f}| "
        f"= {err.max().item():.4f} too large; sigma* = {sigma_star.tolist()}"
    )


def test_bisect_median_tau_preserves_response_signal():
    """With the shared median tau, response strength flows into sigma*: a
    high-gain (strong-responder) target gets a SMALLER sigma* (stronger
    connection), a low-gain target gets a LARGER sigma*.

    This is the bias-fix property: the old per-target-max tau made sigma*
    gain-independent (all targets shared the same sigma*), which collapsed the
    connectivity signal. The median tau restores the correct direction.
    """
    x_obs = torch.randn(32, OBS_LEN, C)
    # 6 targets with gains spanning 5 orders of magnitude. Median gain ≈ 55.
    gain = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    model = _PerturbationSensitiveModel(
        n_channels=C, obs_len=OBS_LEN, fc_len=FC_LEN,
        source_channel=SRC, gain=gain, clean_subject=x_obs,
    )
    sigma_star = bisect_sigma_for_source(
        model, x_obs, source_channel=SRC,
        hparams=default_hparams(alpha=1.0, precision=1e-3, batch_size=8),
    )
    g_sorted_idx = torch.argsort(gain)           # ascending gain
    # Highest-gain target should have the SMALLEST sigma* (strongest conn).
    strongest = g_sorted_idx[-1].item()
    weakest = g_sorted_idx[0].item()
    assert sigma_star[strongest] < sigma_star[weakest], (
        f"high-gain target sigma*={sigma_star[strongest]:.4f} should be < "
        f"low-gain target sigma*={sigma_star[weakest]:.4f}; "
        f"sigma* = {sigma_star.tolist()}, gain = {gain.tolist()}"
    )


def test_bisect_boundary_destabilised_at_min():
    """If sigma_min already exceeds sigma_max*sqrt(alpha) (i.e. the half-max
    point is below sigma_min), every target clamps to sigma_min.

    The min boundary fires when mse_at_min > tau, i.e.
        gain^2 * sigma_min^2 > alpha * gain^2 * sigma_max^2
        => sigma_min > sigma_max * sqrt(alpha).
    With sigma_max=0.3 and alpha=0.5, we need sigma_min > 0.212. We pass
    sigma_min=0.25 to trigger the boundary (independent of gain).
    """
    x_obs = torch.randn(16, OBS_LEN, C)
    model = _PerturbationSensitiveModel(
        n_channels=C, obs_len=OBS_LEN, fc_len=FC_LEN,
        source_channel=SRC, gain=torch.ones(C), clean_subject=x_obs,
    )
    sigma_star = bisect_sigma_for_source(
        model, x_obs, source_channel=SRC,
        hparams=default_hparams(
            alpha=0.5, sigma_min=0.25, sigma_max=0.3,
            precision=1e-3, batch_size=8,
        ),
    )
    assert sigma_star.shape == (C,)
    # All clamped to sigma_min (half-max point 0.212 < 0.25).
    assert torch.allclose(sigma_star, torch.full((C,), 0.25))


def test_bisect_recovers_full_range_when_alpha_high():
    """With alpha close to 1, tau ~ mse_at_max so sigma* -> sigma_max."""
    x_obs = torch.randn(32, OBS_LEN, C)
    model = _PerturbationSensitiveModel(
        n_channels=C, obs_len=OBS_LEN, fc_len=FC_LEN,
        source_channel=SRC, gain=torch.ones(C), clean_subject=x_obs,
    )
    sigma_star = bisect_sigma_for_source(
        model, x_obs, source_channel=SRC,
        hparams=default_hparams(alpha=0.99, precision=1e-3, batch_size=8),
    )
    # sigma* = sigma_max * sqrt(0.99) ~ 0.2985, within precision of sigma_max.
    target = SIGMA_MAX * math.sqrt(0.99)
    err = (sigma_star - target).abs()
    assert err.max().item() < 2e-2


def test_default_hparams_overrides():
    hp = default_hparams(alpha=0.5, precision=5e-4)
    assert hp["alpha"] == 0.5
    assert hp["precision"] == 5e-4
    # Untouched keys keep defaults.
    assert hp["sigma_min"] == SIGMA_MIN
    assert hp["sigma_max"] == SIGMA_MAX
    assert hp["use_proj"] is False
