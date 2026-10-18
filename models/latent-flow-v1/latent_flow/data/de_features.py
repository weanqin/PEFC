"""Differential-entropy (DE) 5-band node features for EEG graph classification.

This module computes per-channel, per-band differential entropy from raw EEG
windows, producing the node-feature matrix ``(C, 5)`` consumed by the DCGNN
classifier. Nothing like this existed in the repo before (the "DE" elsewhere
refers to the Discard-Entropy connectivity probe, a different concept).

Mathematical convention (matches MODMA / SEED / the DCGNN aerospace variant):
    For a band-limited EEG segment modelled as a zero-mean Gaussian with
    variance ``σ²``, the differential entropy is

        DE = 0.5 * log(2πe · σ²)

    We estimate ``σ²`` as the band power (integral of the Welch PSD over the
    band). Because the input EEG is per-channel z-scored (unit variance, see
    ``data/modma.py``), DE values are comparable across channels and subjects.

Bands (Hz, at the project's 250 Hz sampling rate):
    delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45.

Tensor convention follows the repo standard (B, T, C) for windowed data, with
raw subject recordings carried as (C, T) per ``data/modma.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Canonical 5-band edges (Hz). Order is fixed: [delta, theta, alpha, beta, gamma].
BAND_EDGES: list[tuple[float, float]] = [
    (1.0, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 30.0),  # beta
    (30.0, 45.0),  # gamma
]
N_BANDS = len(BAND_EDGES)

# Differential-entropy constant: 0.5 * log(2*pi*e).
_HALF_LOG_2PIE = 0.5 * np.log(2.0 * np.pi * np.e)

# Eps added to band power before the log, to keep DE finite for a silent channel.
_POWER_EPS = 1e-12


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrate PSD over [lo, hi] Hz for every channel.

    Args:
        freqs: (n_freq,) frequency axis from ``welch``.
        psd: (C, n_freq) one-sided PSD per channel.
        lo, hi: band edges (Hz).

    Returns:
        (C,) band power per channel.
    """
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        # Band falls outside the resolved spectrum — return zeros (will map to a
        # very negative DE after the log, which is the correct "no power" signal).
        return np.zeros(psd.shape[0], dtype=psd.dtype)
    # Trapezoidal integration of the one-sided PSD over the band. The frequency
    # spacing is uniform, so a simple mean*df is exact for the trapezoidal rule.
    df = freqs[1] - freqs[0]
    return psd[:, mask].sum(axis=1) * df


def compute_de_features(eeg: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """Compute 5-band differential-entropy node features for one EEG window.

    Args:
        eeg: (T, C) or (C, T) raw EEG window. If the time axis is last (shape
            ``(C, T)`` as returned by ``load_mat_file``), it is detected by the
            ``fs``-plausible axis length and transposed. ``(T, C)`` (the repo's
            window convention from ``extract_windows``) is the canonical input.
        fs: sampling rate in Hz (default 250, the project rate).

    Returns:
        de: (C, 5) differential-entropy features, ordered by ``BAND_EDGES``.
        Finite everywhere (band power is floor-clipped at ``_POWER_EPS``).
    """
    arr = np.asarray(eeg, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"eeg must be 2D (T,C) or (C,T), got shape {arr.shape}")

    # Heuristic axis detection: the time axis is the one whose length is at
    # least ~1s of samples. If both qualify, prefer the longer as time. This
    # mirrors how callers pass either raw (C,T) or windowed (T,C) data.
    t, c = arr.shape
    if c >= t and c > int(fs):
        # Looks like (C, T) — transpose to (T, C).
        arr = arr.T
    n_time, n_chan = arr.shape

    if n_time < 4:
        raise ValueError(
            f"need at least 4 samples for Welch, got n_time={n_time} (fs={fs})"
        )

    # Welch's method: robust PSD estimate along the time axis (axis 0). Returns
    # psd of shape (n_time_seg_freqs, n_chan); transpose to (n_chan, n_freq) for
    # per-channel band integration. nperseg is capped at n_time; the default 256
    # (~1s at 250 Hz) resolves all 5 bands including delta (1 Hz).
    nperseg = min(256, n_time)
    freqs, psd = welch(arr, fs=fs, nperseg=nperseg, axis=0)  # (F,) and (F, C)
    psd = psd.T  # (C, F)

    de = np.empty((n_chan, N_BANDS), dtype=np.float32)
    for b, (lo, hi) in enumerate(BAND_EDGES):
        power = _band_power(freqs, psd, lo, hi)  # (C,)
        de[:, b] = _HALF_LOG_2PIE + 0.5 * np.log(power + _POWER_EPS)
    return de


def aggregate_subject_de(windows: list[np.ndarray], fs: float = 250.0) -> np.ndarray:
    """Mean-aggregate DE features across a subject's windows.

    Args:
        windows: list of (T, C) or (C, T) EEG windows (e.g. from
            ``extract_windows`` transposed, or raw (C,T) slices).
        fs: sampling rate.

    Returns:
        (C, 5) subject-level DE = elementwise mean over windows.
    """
    if not windows:
        raise ValueError("windows list is empty")
    stacked = np.stack([compute_de_features(w, fs=fs) for w in windows], axis=0)
    return stacked.mean(axis=0)
