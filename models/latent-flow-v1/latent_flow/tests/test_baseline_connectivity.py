"""Unit tests for baseline directed-connectivity kernels (task 06-25).

Verifies the GC and symbolic-TE estimators recover known coupling
directions on synthetic data, and are ~0 on independent series. These are
the correctness contracts the fold runner depends on.

Run (conda env npi):

    D:\\program\\anaconda\\envs\\npi\\python.exe -m pytest \\
        tests/test_baseline_connectivity.py -q

All output ASCII (Windows GBK safe, per logging-guidelines.md).
"""

from __future__ import annotations

import numpy as np

from latent_flow.baseline.granger import (
    _spectral_gc_pair,
    granger_connectivity,
)
from latent_flow.baseline.transfer_entropy import (
    _permutation_symbols,
    _symbolic_te,
    transfer_entropy_connectivity,
)

FS = 250.0


# ---------------------------------------------------------------------------
# Synthetic data generators with known coupling
# ---------------------------------------------------------------------------

def _coupled_ar(seed=0, n=6000, c_ab=0.4):
    """Bivariate AR(1) with unidirectional coupling a -> b.

    a[t] = 0.6 a[t-1] + e_a
    b[t] = 0.5 b[t-1] + c_ab * a[t-1] + e_b
    """
    rng = np.random.RandomState(seed)
    a = np.zeros(n); b = np.zeros(n)
    for t in range(2, n):
        a[t] = 0.6 * a[t - 1] + rng.randn()
        b[t] = 0.5 * b[t - 1] + c_ab * a[t - 1] + rng.randn()
    return a, b


def _independent(seed=0, n=6000):
    """Two independent white-noise series (no coupling)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n), rng.randn(n)


# ---------------------------------------------------------------------------
# Granger Causality tests
# ---------------------------------------------------------------------------

def test_gc_recovers_coupling_direction():
    """GC a->b must be much larger than b->a for unidirectional a->b."""
    a, b = _coupled_ar(c_ab=0.4)
    gc_ab, gc_ba = _spectral_gc_pair(a, b, FS, max_order=8, n_freq=65,
                                     fmin=1.0, fmax=125.0)
    assert gc_ab > 5.0, f"a->b GC too small: {gc_ab}"
    assert gc_ab > 5 * gc_ba, f"directionality weak: {gc_ab} vs {gc_ba}"


def test_gc_zero_on_independent():
    """GC on independent series must be ~0 (no false coupling)."""
    c, d = _independent()
    gc_cd, gc_dc = _spectral_gc_pair(c, d, FS, max_order=8, n_freq=65,
                                     fmin=1.0, fmax=125.0)
    assert gc_cd < 0.2, f"indep GC c->d too large: {gc_cd}"
    assert gc_dc < 0.2, f"indep GC d->c too large: {gc_dc}"


def test_gc_matrix_shape_and_diagonal():
    """granger_connectivity returns (n, n) with zero diagonal."""
    a, b = _coupled_ar()
    c, d = _independent()
    data = np.stack([a, b, c, d])
    conn = granger_connectivity(data, fs=FS)
    assert conn.shape == (4, 4)
    assert np.all(np.diag(conn) == 0.0)
    # a->b is the strongest directed edge (rows = source).
    assert conn[0, 1] > conn[1, 0]


# ---------------------------------------------------------------------------
# Symbolic Transfer Entropy tests
# ---------------------------------------------------------------------------

def test_permutation_symbols_alphabet():
    """m=3 yields symbols in [0, 6) and correct length."""
    rng = np.random.RandomState(0)
    s = rng.randn(1000)
    sym = _permutation_symbols(s, m=3, tau=1)
    assert sym.shape == (1000 - 2,)   # n - (m-1)*tau
    assert sym.min() >= 0 and sym.max() < 6


def test_te_recovers_coupling_delayed_copy():
    """TE a->b large, b->a ~0 for a perfect delayed copy b[t]=a[t-1].

    Symbolic TE has a small positive finite-sample bias, so the bar for the
    reverse direction is a few times the independent noise level (~1e-3),
    not strict zero.
    """
    rng = np.random.RandomState(0)
    a = rng.randn(20000)
    b = np.roll(a, 1); b[0] = 0.0
    sa = _permutation_symbols(a); sb = _permutation_symbols(b)
    te_ab = _symbolic_te(sa, sb)
    te_ba = _symbolic_te(sb, sa)
    assert te_ab > 0.5, f"a->b TE too small: {te_ab}"
    assert te_ab > 10 * te_ba, f"directionality weak: {te_ab} vs {te_ba}"


def test_te_low_on_independent():
    """TE on independent series is near the finite-sample bias (~1e-3)."""
    c, d = _independent()
    te_cd = _symbolic_te(_permutation_symbols(c), _permutation_symbols(d))
    assert te_cd < 0.01, f"indep TE too large: {te_cd}"


def test_te_matrix_shape_and_diagonal():
    """transfer_entropy_connectivity returns (n, n) with zero diagonal."""
    a, b = _coupled_ar()
    c, d = _independent()
    data = np.stack([a, b, c, d])
    conn = transfer_entropy_connectivity(data)
    assert conn.shape == (4, 4)
    assert np.all(np.diag(conn) == 0.0)
    assert conn[0, 1] >= 0.0 and np.isfinite(conn).all()
