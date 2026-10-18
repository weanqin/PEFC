"""Unit tests for baseline/gpdc.py — pairwise bivariate gPDC.

Tests:
    1. Shape sanity: 128x128 directed matrix, diag=0, finite values.
    2. Synthetic 4-ch VAR recovery: known coupling → gPDC identifies direction.
    3. default_params returns a dict with expected keys.
    4. Pairwise helper produces expected shape.
"""
import numpy as np
import pytest

from latent_flow.baseline.gpdc import (
    gpdc_connectivity,
    default_params,
    _pairwise_gpdc,
)


def generate_var4_data(n_samples=5000, seed=42):
    """Generate 4-channel VAR(2) data with known directed coupling.

    VAR model: X[t] = A1 @ X[t-1] + A2 @ X[t-2] + eps[t], eps ~ N(0, 0.01*I).
    Convention: A_k[i,j] = coefficient from x_j[t-k] to x_i[t] (j→i).

    Topology (row=target, col=source — j→i convention):
        ch0 → ch1:  A1[1,0] = 0.4,  A2[1,0] = 0.1
        ch1 → ch2:  A1[2,1] = 0.35
        ch2 → ch3:  A1[3,2] = 0.3
        No reverse: A1[0,1]=A1[1,2]=A1[2,3]=0.
        Self-coupling (diagonal) at ~0.6 for stability.
    """
    rng = np.random.RandomState(seed)
    C = 4
    p = 2

    A1 = np.zeros((C, C))
    A1[0, 0] = 0.6
    A1[1, 1] = 0.55
    A1[2, 2] = 0.5
    A1[3, 3] = 0.45
    A1[1, 0] = 0.4   # 0→1
    A1[2, 1] = 0.35  # 1→2
    A1[3, 2] = 0.3   # 2→3

    A2 = np.zeros((C, C))
    A2[0, 0] = 0.1
    A2[1, 1] = 0.1
    A2[2, 2] = 0.1
    A2[3, 3] = 0.1
    A2[1, 0] = 0.1   # 0→1 (2nd order)
    A2[3, 0] = 0.1   # 0→3 (2nd order)

    burn_in = 500
    total = burn_in + n_samples
    X = np.zeros((C, total))
    for t in range(p, total):
        eps = rng.randn(C) * 0.1
        X[:, t] = A1 @ X[:, t - 1] + A2 @ X[:, t - 2] + eps

    return X[:, burn_in:]  # (4, n_samples)


class TestGPDSmoke:
    """Smoke tests for pairwise gPDC kernel."""

    def test_shape_and_diag(self):
        """128x128 white noise → finite 128x128 matrix, diag=0."""
        rng = np.random.RandomState(99)
        data = rng.randn(128, 2000).astype(np.float32)
        conn = gpdc_connectivity(data, fs=250.0, max_order=5, n_freq=25,
                                  fmin=1.0, fmax=125.0)
        assert conn.shape == (128, 128)
        assert conn.dtype == np.float32
        assert np.all(np.isfinite(conn))
        assert np.allclose(np.diag(conn), 0.0)

    def test_default_params(self):
        p = default_params()
        assert p["method"] == "gpdc_pairwise_bivariate"
        assert "max_order" in p
        assert "fs" in p

    def test_directedness_nonzero(self):
        """On white noise, pairwise gPDC should produce some nonzero values."""
        rng = np.random.RandomState(42)
        data = rng.randn(16, 2000).astype(np.float32)
        conn = gpdc_connectivity(data, fs=250.0, max_order=8, n_freq=25,
                                  fmin=1.0, fmax=125.0)
        offdiag = conn[~np.eye(16, dtype=bool)]
        assert np.any(offdiag > 0), "All off-diagonal gPDC values are zero"


class TestVARRecovery:
    """Synthetic VAR ground-truth recovery test."""

    def test_direction_recovery(self):
        """Known 0→1, 1→2, 2→3 couplings → gPDC identifies direction."""
        data = generate_var4_data(n_samples=10000, seed=123)
        conn = gpdc_connectivity(data, fs=250.0, max_order=8, n_freq=65,
                                  fmin=1.0, fmax=125.0)

        assert conn[0, 1] > conn[1, 0], \
            f"0→1 ({conn[0,1]:.4f}) should exceed 1→0 ({conn[1,0]:.4f})"
        assert conn[1, 2] > conn[2, 1], \
            f"1→2 ({conn[1,2]:.4f}) should exceed 2→1 ({conn[2,1]:.4f})"
        assert conn[2, 3] > conn[3, 2], \
            f"2→3 ({conn[2,3]:.4f}) should exceed 3→2 ({conn[3,2]:.4f})"

    def test_false_positive_rate(self):
        """Reverse edges should be much weaker than forward edges."""
        data = generate_var4_data(n_samples=10000, seed=456)
        conn = gpdc_connectivity(data, fs=250.0, max_order=8, n_freq=65,
                                  fmin=1.0, fmax=125.0)

        assert conn[1, 0] < 0.1 * conn[0, 1], \
            f"1→0 ({conn[1,0]:.4f}) should be ≪ 0→1 ({conn[0,1]:.4f})"
        assert conn[2, 1] < 0.1 * conn[1, 2], \
            f"2→1 ({conn[2,1]:.4f}) should be ≪ 1→2 ({conn[1,2]:.4f})"
        assert conn[3, 2] < 0.25 * conn[2, 3], \
            f"3→2 ({conn[3,2]:.4f}) should be ≪ 2→3 ({conn[2,3]:.4f})"


class TestPairwiseHelper:
    """Unit tests for the per-pair gPDC helper."""

    def test_pairwise_returns_finite(self):
        """_pairwise_gpdc returns two finite floats on white noise."""
        rng = np.random.RandomState(1)
        x = rng.randn(5000)
        y = rng.randn(5000)
        xy, yx = _pairwise_gpdc(x, y, fs=250.0, max_order=10,
                                 n_freq=33, fmin=1.0, fmax=125.0)
        assert np.isfinite(xy)
        assert np.isfinite(yx)
        assert xy > 0
        assert yx > 0

    def test_pairwise_nice_values(self):
        """On Gaussian noise, gPDC values are in a sensible range (0, 1)."""
        rng = np.random.RandomState(2)
        x = rng.randn(5000)
        y = rng.randn(5000)
        xy, yx = _pairwise_gpdc(x, y, fs=250.0, max_order=10,
                                 n_freq=33, fmin=1.0, fmax=125.0)
        # gPDC is normalized and should be in [0, 1]
        assert 0 <= xy <= 1, f"gPDC out of range: {xy:.4f}"
        assert 0 <= yx <= 1, f"gPDC out of range: {yx:.4f}"
