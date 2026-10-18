"""Multi-metric forecast evaluation helpers for MODMA EEG.

Extracted from the former ``scripts/eval_model_compare.py`` so the OC-LatentFlow
eval/diag pipeline can reuse these without importing the (now-archived) JET /
latent-flow comparison script.

Per ``domain.md`` "Evaluation Metrics", model comparison MUST report the full
panel (MAE + Pearson + PSD-L1 + std_ratio + diversity + coherence): MAE alone
is gamed by output collapse. A model only counts as "generating real
waveforms" if ``std_ratio >= 0.5`` AND ``PSD-L1 <= 1.0``.

All functions take ``(B, T, C)`` tensors on any device; numpy/PSD work is done
on CPU. Aggregation is sample-weighted so batches of unequal size combine
correctly.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.signal import coherence, welch

# MODMA sampling rate (Hz). Used as the default for the PSD/coherence helpers
# so callers don't have to pass it every time.
FS = 250.0


# =============================================================================
# Metric helpers
# =============================================================================

def per_channel_pearson(x_hat, x_fc, eps=1e-8):
    """Per-channel Pearson r, mean over channels.

    Args:
        x_hat, x_fc: (B, T, C)
    Returns:
        scalar mean Pearson r across channels and batch
    """
    mu_h = x_hat.mean(dim=1, keepdim=True)
    mu_t = x_fc.mean(dim=1, keepdim=True)
    h = x_hat - mu_h
    t = x_fc - mu_t
    cov = (h * t).sum(dim=1)            # (B, C)
    nrm = h.norm(dim=1) * t.norm(dim=1) + eps
    r = cov / nrm
    return r.mean().item()


def log_psd_l1(x_hat, x_fc, fs=FS):
    """Mean L1 distance between log-PSDs averaged over channels.

    Args:
        x_hat, x_fc: (B, T, C)
    Returns:
        scalar mean L1 distance
    """
    B, T, C = x_hat.shape
    nperseg = min(T, 64)
    h_np = x_hat.cpu().numpy()
    t_np = x_fc.cpu().numpy()
    total = 0.0
    count = 0
    # Loop over channels only (vectorize over batch via axis=-1).
    for c in range(C):
        _, p_t = welch(t_np[:, :, c], fs=fs, nperseg=nperseg, axis=-1)
        _, p_h = welch(h_np[:, :, c], fs=fs, nperseg=nperseg, axis=-1)
        p_t = np.clip(p_t, 1e-12, None)
        p_h = np.clip(p_h, 1e-12, None)
        total += np.abs(np.log10(p_t) - np.log10(p_h)).mean()
        count += 1
    return float(total / max(count, 1))


def spectral_coherence(x_hat, x_fc, fs=FS, band=(1.0, 40.0)):
    """Mean magnitude-squared coherence in the 1-40 Hz band, per channel.

    Args:
        x_hat, x_fc: (B, T, C)
    Returns:
        scalar mean coherence (NaN if prediction has zero variance, e.g. zero baseline)
    """
    B, T, C = x_hat.shape
    nperseg = min(T, 64)
    h_np = x_hat.cpu().numpy()
    t_np = x_fc.cpu().numpy()
    coh_sum = 0.0
    count = 0
    for c in range(C):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f, coh = coherence(
                t_np[:, :, c], h_np[:, :, c],
                fs=fs, nperseg=nperseg, axis=-1,
            )
            mask = (f >= band[0]) & (f <= band[1])
            if mask.sum() > 0:
                with np.errstate(invalid="ignore"):
                    m = np.nanmean(coh[:, mask])
                if np.isfinite(m):
                    coh_sum += m
                    count += 1
    if count == 0:
        return float("nan")
    return float(coh_sum / count)


def within_batch_diversity(x_hat, eps=1e-8, max_pairs=64):
    """Mean Pearson r between distinct samples in batch (high = collapse).

    To stay efficient we cap the number of pairwise comparisons.
    """
    B = x_hat.shape[0]
    if B < 2:
        return 0.0
    # Flatten (T, C) per sample.
    flat = x_hat.reshape(B, -1)
    flat_c = flat - flat.mean(dim=1, keepdim=True)
    rng = np.random.RandomState(0)
    pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    rs = []
    for i, j in pairs:
        a, b = flat_c[i], flat_c[j]
        denom = a.norm() * b.norm() + eps
        rs.append(((a * b).sum() / denom).item())
    return float(np.mean(rs))


# =============================================================================
# Aggregator
# =============================================================================

class MetricAggregator:
    """Accumulates per-batch metrics, sample-weighted."""

    def __init__(self):
        self.sums = {
            "mae": 0.0,
            "pearson": 0.0,
            "psd_l1": 0.0,
            "coherence": 0.0,
            "diversity": 0.0,
        }
        self.std_sum = 0.0
        self.std_truth_sum = 0.0
        self.n = 0

    def update(self, x_hat, x_fc):
        B = x_fc.shape[0]
        self.sums["mae"] += (x_hat - x_fc).abs().mean().item() * B
        self.sums["pearson"] += per_channel_pearson(x_hat, x_fc) * B
        self.sums["psd_l1"] += log_psd_l1(x_hat, x_fc) * B
        self.sums["coherence"] += spectral_coherence(x_hat, x_fc) * B
        self.sums["diversity"] += within_batch_diversity(x_hat) * B
        # Std computed over time axis (per sample, per channel), then averaged.
        self.std_sum += x_hat.std(dim=1).mean().item() * B
        self.std_truth_sum += x_fc.std(dim=1).mean().item() * B
        self.n += B

    def summary(self):
        if self.n == 0:
            return {k: float("nan") for k in self.sums} | {"std_ratio": float("nan")}
        out = {k: v / self.n for k, v in self.sums.items()}
        out["std_ratio"] = self.std_sum / max(self.std_truth_sum, 1e-12)
        return out
