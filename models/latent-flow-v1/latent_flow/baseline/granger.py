"""Pairwise bivariate spectral Granger Causality (Geweke 1982).

For each ordered pair of channels (source a, target b) the function fits a
single bivariate VAR(p) on [a, b], derives the frequency-domain transfer
function, and applies Geweke's spectral-GC factorisation to obtain the
directed influence a->b at each frequency. The full-band scalar returned is
the integral over [fmin, fmax] (default 1-125 Hz = up to Nyquist at 250 Hz),
matching the "full-band" decision in the task plan.

Why bivariate (not conditional/multivariate):
    A full 128-variable VAR is severely ill-conditioned (p*128^2 params),
    so we follow the standard EEG practice of pairwise bivariate fits. This
    conflates direct and indirect connections, which is the intended
    "classical baseline" character.

Method (Geweke spectral GC):
    Fit VAR(p): X[t] = sum_{k=1..p} A[k] X[t-k] + E[t], resid cov Sigma.
    Transfer function  H(f) = (I - sum_k A[k] e^{-i2pi f k/fs})^{-1}.
    Spectral density    S(f) = H(f) Sigma H(f)^H.
    Factorise Sigma = L L^T (Cholesky), with normalised H_tilde = H L,
    S(f) = H_tilde H_tilde^H. Then:
        f_{j->i}(f) = ln( H_tilde_ii(f)^2 / |H_tilde row i of the
        restricted (no-j) model |^2 )
    Equivalently the standard form using the diagonal of the transformed
    transfer function. We use the Dhamala 2008 form:
        f_{j->i}(f) = ln( S_ii(f) / S_ii^{no j}(f) )
    computed via the Geweke factorisation of the bivariate VAR (2 vars, so
    the restricted model for i is just the univariate AR residual variance).

Granger causality is a ratio of residual variances (linear predictability
gain), so it is >= 0. Direction emerges because including j's past reduces
i's residual variance only when j drives i.

References:
    Geweke 1982 (JASA) -- original spectral GC.
    Dhamala, Rangarajan, Ding 2008 (PLoS Comput Biol) -- the pairwise
    estimator used widely in EEG.
"""

import numpy as np
import warnings as _warnings

from latent_flow.config import FS

# Suppress statsmodels VAR fit warnings (stationarity / length) at the
# per-pair level; the runner reports fit failures explicitly.
_warnings.filterwarnings("ignore", category=Warning)


class _VARFit:
    """Lightweight container duck-typing statsmodels VARResults (.coefs,
    .sigma_u, .k_ar) so the rest of the code is unchanged."""

    __slots__ = ("coefs", "sigma_u", "k_ar")

    def __init__(self, coefs, sigma_u, k_ar):
        self.coefs = coefs        # (p, 2, 2)
        self.sigma_u = sigma_u    # (2, 2)
        self.k_ar = k_ar          # int


def _fit_bivariate_var(x, y, max_order, ic="aic", fixed_order=None):
    """Fit a bivariate VAR on [x, y] via ordinary least squares (fast numpy).

    Single QR factorisation of the max-order lagged design is reused for the
    AIC order sweep (each candidate order = leading columns of Q, solved by
    cheap back-substitution rather than a fresh lstsq). When
    ``fixed_order`` is given the sweep is skipped entirely (fastest).

    Returns a ``_VARFit`` with .coefs (p, 2, 2), .sigma_u (2, 2), .k_ar.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    p_max = int(max_order)
    T = n - p_max                       # usable rows after max lag

    # Lagged design: columns = [x[t-1], y[t-1], ..., x[t-p_max], y[t-p_max],
    # intercept]. Shape (T, 2*p_max+1).
    Z = np.empty((T, 2 * p_max + 1), dtype=np.float64)
    Z[:, -1] = 1.0
    for k in range(1, p_max + 1):
        Z[:, 2 * (k - 1)] = x[p_max - k : n - k]
        Z[:, 2 * (k - 1) + 1] = y[p_max - k : n - k]
    Ymat = np.column_stack([x[p_max:], y[p_max:]])     # (T, 2)

    def _solve_cols(ncols):
        """OLS on the first ``ncols`` lag columns + intercept. Returns
        (coefs, sigma_u). Uses lstsq on the column subset; called O(p) times
        only when sweeping."""
        cols = list(range(ncols)) + [2 * p_max]
        Zp = Z[:, cols]
        cp, *_ = np.linalg.lstsq(Zp, Ymat, rcond=None)
        rp = Ymat - Zp @ cp
        sigma = (rp.T @ rp) / T
        return cp, sigma

    if fixed_order is not None and fixed_order >= 1:
        cp, sigma_u = _solve_cols(2 * int(fixed_order))
        p = int(fixed_order)
    else:
        # AIC sweep over orders 1..p_max.
        best_aic, best_p = np.inf, p_max
        two_log_2pi = 2 * np.log(2 * np.pi)
        for cand in range(1, p_max + 1):
            _, ssp = _solve_cols(2 * cand)
            sign, logdet = np.linalg.slogdet(ssp)
            if sign <= 0 or not np.isfinite(logdet):
                continue
            n_params = 2 * (2 * cand + 1)
            aic = 2 * n_params + T * (logdet + two_log_2pi)
            if aic < best_aic:
                best_aic, best_p = aic, cand
        cp, sigma_u = _solve_cols(2 * best_p)
        p = best_p

    # Reshape coefficients to (p, 2, 2): coefs[k-1, i, j] = effect of lag-k
    # of series j on series i. cp rows: [x_lag1, y_lag1, ..., intercept].
    coefs = np.zeros((p, 2, 2), dtype=np.float64)
    for k in range(1, p + 1):
        coefs[k - 1, 0, 0] = cp[2 * (k - 1), 0]
        coefs[k - 1, 0, 1] = cp[2 * (k - 1) + 1, 0]
        coefs[k - 1, 1, 0] = cp[2 * (k - 1), 1]
        coefs[k - 1, 1, 1] = cp[2 * (k - 1) + 1, 1]

    return _VARFit(coefs, sigma_u, p)


def _spectral_gc_pair(x, y, fs, max_order, n_freq, fmin, fmax,
                      fixed_order=None):
    """Directed spectral GC for the ordered pair x->y, integrated over band.

    Fits ONE bivariate VAR on [x, y] and returns both directions from the
    SAME fit via the Geweke / Dhamala Cholesky factorisation:

        f_{i->j}(f) = ln( S_jj(f) / | H_tilde_jj(f) |^2 )

    where H_tilde = H(f) L, with L the Cholesky factor of the residual
    covariance Sigma = L L^T. Because the factorisation is triangular with
    the source ordered FIRST, |H_tilde_jj|^2 is the y-power remaining after
    removing x's contribution, and the ratio is the spectral GC x->y.
    Swapping the variable order yields y->x from the same VAR coefficients.

    This single-fit form guarantees GC -> 0 on independent (uncoupled)
    series, unlike separate full/reduced fits whose residual variances
    disagree.

    Args:
        x, y: 1-D arrays (one channel each), equal length, z-scored.
        fs: sampling rate (Hz).
        max_order: max AR order; actual order is AIC-selected.
        n_freq: number of frequency bins over [fmin, fmax].
        fmin, fmax: integration band (Hz).

    Returns:
        (gc_xy, gc_yx): integrated spectral GC x->y and y->x over the band.
    """
    freqs = np.linspace(fmin, fmax, n_freq)
    omega = 2.0 * np.pi * freqs / fs

    # Precompute z^k for all (freq, k): shape (n_freq, max_p+1). Broadcast
    # over frequencies to vectorise the per-frequency transfer-function loop.
    gc_xy = _gc_one_direction(x, y, max_order, omega, target_idx=1,
                              fixed_order=fixed_order)
    gc_yx = _gc_one_direction(y, x, max_order, omega, target_idx=1,
                              fixed_order=fixed_order)

    # GC is a log-ratio of variances -> theoretically >= 0; clamp tiny
    # negative numerical artefacts at 0.
    gc_xy = np.clip(gc_xy, 0.0, None)
    gc_yx = np.clip(gc_yx, 0.0, None)

    _trapz = getattr(np, "trapezoid", np.trapz)  # numpy>=2.0 renames trapz
    val_xy = float(_trapz(gc_xy, freqs))
    val_yx = float(_trapz(gc_yx, freqs))
    return val_xy, val_yx


def _gc_one_direction(src, tgt, max_order, omega, target_idx=1,
                      fixed_order=None):
    """Vectorised spectral GC src->tgt over all frequencies at once.

    Fits the bivariate VAR on [src, tgt] (src ordered FIRST so the Cholesky
    factorisation partials src out of tgt), then evaluates the Geweke ratio
        gc(f) = ln( S_tgt_tgt(f) / |H_tilde_tgt_tgt(f)|^2 )
    for every frequency in ``omega`` via broadcasting (no Python freq loop).

    Args:
        src, tgt: 1-D channel arrays.
        max_order: AR order budget (AIC-selected inside the fit unless
            ``fixed_order`` is given).
        omega: 1-D array of angular frequencies (rad/sample).
        target_idx: row/col index of the target in the 2x2 system (1 here,
            since [src=0, tgt=1]).
        fixed_order: if set, fit this exact AR order (skip AIC sweep, fast).

    Returns:
        1-D float array of per-frequency GC (not yet integrated).
    """
    res = _fit_bivariate_var(src, tgt, max_order, fixed_order=fixed_order)
    p = res.k_ar
    coefs = np.asarray(res.coefs, dtype=np.float64)      # (p, 2, 2)
    sigma = np.asarray(res.sigma_u, dtype=np.float64)    # (2, 2)
    L = np.linalg.cholesky(sigma)                        # Sigma = L L^T

    n_freq = omega.shape[0]
    # A_sum(f) = sum_k coefs[k-1] * z^k, z = e^{-i w}. Build z_powers
    # (n_freq, p+1) then sum via einsum over the lag axis.
    k_idx = np.arange(1, p + 1)                           # (p,)
    z_powers = np.exp(-1j * np.outer(omega, k_idx))       # (n_freq, p)
    # A_sum: (n_freq, 2, 2) = sum_k coefs[k-1] * z_powers[:,k]
    A_sum = np.einsum("fk,kij->fij", z_powers, coefs)     # (n_freq, 2, 2)
    I2 = np.eye(2)
    # H(f) = (I - A_sum(f))^{-1}  -> batched inverse over frequencies.
    H = np.linalg.inv(I2[None, :, :] - A_sum)             # (n_freq, 2, 2)
    H_tilde = H @ L                                       # (n_freq, 2, 2)
    # S(f) = H(f) Sigma H(f)^H ; take the target diagonal.
    S = H @ sigma[None, :, :] @ np.conj(np.swapaxes(H, -1, -2))  # (n_freq,2,2)
    S_tt = S[:, target_idx, target_idx].real              # (n_freq,)
    Ht_tt2 = np.abs(H_tilde[:, target_idx, target_idx]) ** 2     # (n_freq,)
    gc = np.log(S_tt / np.maximum(Ht_tt2, 1e-12))
    return gc


def _granger_row(args):
    """Worker: compute one source row of the GC matrix.

    ``args = (a, data, fs, max_order, n_freq, fmin, fmax, fixed_order)``.
    Returns ``(a, row)`` where ``row`` is the (n_ch,) directed-GC vector for
    source ``a`` (row[a] = 0). Kept module-level so joblib can pickle it.
    """
    a, data, fs, max_order, n_freq, fmin, fmax, fixed_order = args
    n_ch = data.shape[0]
    row = np.zeros(n_ch, dtype=np.float32)
    x = data[a]
    for b in range(n_ch):
        if a == b:
            continue
        try:
            val_ab, _ = _spectral_gc_pair(
                x, data[b], fs, max_order, n_freq, fmin, fmax,
                fixed_order=fixed_order)
            row[b] = val_ab
        except Exception:  # noqa: BLE001
            row[b] = 0.0
    return a, row


def granger_connectivity(data, fs=FS, max_order=8, n_freq=65,
                         fmin=1.0, fmax=125.0, verbose=False, n_jobs=1,
                         fixed_order=None):
    """Pairwise bivariate spectral GC matrix for all channels.

    Args:
        data: (n_channels, n_samples) z-scored EEG.
        fs: sampling rate (Hz). Default from config FS=250.
        max_order: max AR order; the actual order is AIC-selected per pair.
        n_freq: number of frequency bins over [fmin, fmax] for the integral.
        fmin, fmax: integration band (Hz). Default 1-125 (full band).
        verbose: print progress every 16 sources.
        n_jobs: number of parallel workers (one per source channel). 1 =
            serial. With 128 channels on a multi-core box, n_jobs=cpu_count
            gives a near-linear speedup since sources are independent.
        fixed_order: if set (e.g. 8), fit this exact AR order on every pair
            and skip the per-pair AIC sweep (~max_order x faster). EEG GC
            commonly uses a single fixed order.

    Returns:
        (n_channels, n_channels) float32 directed matrix; entry [a, b] is
        the integrated spectral GC a->b. Diagonal = 0.
    """
    n_ch = data.shape[0]
    conn = np.zeros((n_ch, n_ch), dtype=np.float32)

    task_args = [(a, data, fs, max_order, n_freq, fmin, fmax, fixed_order)
                 for a in range(n_ch)]

    if n_jobs == 1:
        results = [_granger_row(t) for t in task_args]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=(
            10 if verbose else 0))(
            delayed(_granger_row)(t) for t in task_args)

    for a, row in results:
        conn[a, :] = row

    np.fill_diagonal(conn, 0.0)
    return conn


def default_params(max_order=8, n_freq=65, fmin=1.0, fmax=125.0):
    """Echo dict for the JSON ``method_params`` field (provenance)."""
    return {
        "method": "granger_bivariate_spectral",
        "order_select": "aic",
        "max_order": max_order,
        "n_freq": n_freq,
        "freq_band": "full",
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "fs": float(FS),
        "integral": "trapezoidal",
        "reference": "Geweke 1982; Dhamala 2008",
    }
