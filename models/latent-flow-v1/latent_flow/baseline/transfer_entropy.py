"""Symbolic Transfer Entropy (Bandt-Pompe permutation patterns).

For each ordered pair of channels (source a, target b) we:

1. Symbolise each channel's timeseries via the Bandt-Pompe permutation
   pattern of its embedding vectors: with embedding dimension m and delay
   tau, each length-m window maps to the rank order of its values
   (e.g. m=3 -> one of 3! = 6 ordinal patterns). This is the standard
   ``ordp`` symbolisation (Bandt & Pompe 2002).
2. Estimate the symbolic Transfer Entropy from a to b with delay 1 step:
       TE_{a->b} = sum p(s_b(t), s_b(t-1), s_a(t-1))
                       * log[ p(s_b(t) | s_b(t-1), s_a(t-1))
                             / p(s_b(t) | s_b(t-1)) ]
   i.e. how much the source's past symbol reduces uncertainty about the
   target's current symbol beyond the target's own past (Schreiber 2000).

Symbolic TE is model-free (nonlinear), robust to noise, and fast: the
symbol alphabet is tiny (m! symbols), so joint distributions are
well-sampled even on a single subject's ~42k samples. m=3, tau=1 is the
recommended default for resting-state EEG.

Returns a directed 128x128 matrix; entry [a, b] = TE_{a->b}. Diagonal = 0.

References:
    Schreiber 2000 (Phys Rev Lett) -- original Transfer Entropy.
    Bandt & Pompe 2002 (Phys Rev Lett) -- permutation entropy / symbols.
    Staniek & Lehnertz 2008 (Phys Rev Lett) -- symbolic TE in EEG.
"""

import math

import numpy as np

from latent_flow.config import FS


def _permutation_symbols(series, m=3, tau=1):
    """Bandt-Pompe permutation symbols for a 1-D series.

    For each embedding vector [x[t], x[t-tau], ..., x[t-(m-1)tau]] (time
    ordered from latest to earliest), the symbol is the rank pattern of its
    components. Returns an int array of symbol codes (one per valid t).

    Args:
        series: 1-D array.
        m: embedding dimension (pattern length). m=3 -> 6 symbols.
        tau: embedding delay (samples).

    Returns:
        1-D int array of length len(series) - (m-1)*tau, values in [0, m!).
    """
    n = len(series)
    L = n - (m - 1) * tau
    if L <= 0:
        return np.empty(0, dtype=np.int64)

    # Build embedding matrix: rows are embedding vectors, columns are
    # [x[t], x[t-tau], ..., x[t-(m-1)tau]] (newest first).
    emb = np.empty((L, m), dtype=np.float64)
    for j in range(m):
        emb[:, j] = series[(m - 1 - j) * tau : (m - 1 - j) * tau + L]

    # Rank of each component within its embedding vector -> argsort gives the
    # indices that would sort the row ascending; that index tuple is the
    # permutation pattern. We map each distinct permutation to a unique code
    # in [0, m!) via a lookup table keyed on the lexicographic rank of the
    # permutation among all m! orderings.
    order = np.argsort(emb, axis=1, kind="stable")  # (L, m), values in [0,m)

    # Build permutation -> rank lookup. permutations() yields in lex order,
    # so enumerate assigns the lexicographic rank (0..m!-1).
    import itertools
    rank_of = {}
    for r, perm in enumerate(itertools.permutations(range(m))):
        rank_of[perm] = r

    # Hash each row's permutation tuple to its rank. Packing the m indices
    # into a single Python tuple per row; the count of distinct rows is tiny
    # (<= m!), so a dict comprehension is cheap.
    rows = [tuple(int(v) for v in order[i]) for i in range(L)]
    symbols = np.fromiter((rank_of[r] for r in rows), dtype=np.int64, count=L)
    return symbols


def _symbolic_te(sa, sb, m=3, tau=1, lag=1):
    """Symbolic Transfer Entropy sa -> sb.

    TE_{a->b} = sum_{x,y,z} p(x,y,z)
                    * log2[ p(x | y,z) / p(x | y) ]
    where
        x = s_b(t)        (target current symbol)
        y = s_b(t-lag)    (target past symbol)
        z = s_a(t-lag)    (source past symbol)

    Uses base-2 logarithm (result in bits). Lag is in symbol-steps (default
    1 = one symbol delay); the embedding tau is applied inside symbolisation.

    Args:
        sa, sb: permutation-symbol sequences (from _permutation_symbols).
        m: embedding dimension used for symbolisation (determines alphabet
            size m!).
        tau: embedding delay (samples) used for symbolisation.
        lag: symbol-step lag between source/target past and target present.

    Returns:
        float >= 0, symbolic TE in bits.
    """
    n_sym = len(sa)
    if n_sym <= lag + 1:
        return 0.0
    nsym = m ** 0
    # Use only the aligned overlapping portion.
    x = sb[lag:]            # target present
    y = sb[: n_sym - lag]   # target past
    z = sa[: n_sym - lag]   # source past
    n = len(x)

    # Flatten the (x, y, z) triplets and count joint + marginal frequencies.
    # Encode each combination as a single key: x * (K*K) + y * K + z.
    K = int(math.factorial(m))   # alphabet size
    key = (x.astype(np.int64) * K + y.astype(np.int64)) * K + z.astype(np.int64)
    # joint p(x,y,z)
    uniq, counts = np.unique(key, return_counts=True)
    p_xyz = counts / n
    # marginals
    p_yz_counts = np.unique((y.astype(np.int64) * K + z.astype(np.int64)),
                            return_counts=True)
    p_yz = {int(k): c / n for k, c in zip(*p_yz_counts)}
    p_y_counts = np.unique(y, return_counts=True)
    p_y = {int(k): c / n for k, c in zip(*p_y_counts)}
    p_xy_counts = np.unique((x.astype(np.int64) * K + y.astype(np.int64)),
                            return_counts=True)
    p_xy = {int(k): c / n for k, c in zip(*p_xy_counts)}

    te = 0.0
    for key_val, p in zip(uniq, p_xyz):
        xv = key_val // (K * K)
        yz = key_val % (K * K)
        yv = yz // K
        zv = yz % K
        p_yz_val = p_yz.get(int(yz), 0.0)
        p_y_val = p_y.get(int(yv), 0.0)
        xy_key = int(xv) * K + int(yv)
        p_xy_val = p_xy.get(xy_key, 0.0)
        if p_yz_val > 0 and p_y_val > 0 and p_xy_val > 0:
            # p(x|y,z) = p(x,y,z)/p(y,z); p(x|y) = p(x,y)/p(y)
            pcond_yz = p / p_yz_val
            pcond_y = p_xy_val / p_y_val
            te += p * np.log2(pcond_yz / pcond_y)
    return float(max(te, 0.0))


def transfer_entropy_connectivity(data, m=3, tau=1, lag=1, verbose=False):
    """Pairwise symbolic TE matrix for all channels.

    Args:
        data: (n_channels, n_samples) z-scored EEG.
        m: Bandt-Pompe embedding dimension (default 3 -> 6 symbols).
        tau: embedding delay in samples (default 1).
        lag: symbol-step lag for the TE conditional (default 1).
        verbose: print progress every 16 sources.

    Returns:
        (n_channels, n_channels) float32 directed matrix; [a, b] = TE a->b
        (bits). Diagonal = 0.
    """
    n_ch = data.shape[0]
    # Precompute symbols per channel (reused across all pairs).
    symbols = [_permutation_symbols(data[c], m=m, tau=tau) for c in range(n_ch)]

    conn = np.zeros((n_ch, n_ch), dtype=np.float32)
    for a in range(n_ch):
        sa = symbols[a]
        for b in range(n_ch):
            if a == b:
                continue
            conn[a, b] = _symbolic_te(sa, symbols[b], m=m, tau=tau, lag=lag)
        if verbose and (a % 16 == 0 or a == n_ch - 1):
            print(f"  TE source {a+1}/{n_ch} done")

    np.fill_diagonal(conn, 0.0)
    return conn


def default_params(m=3, tau=1, lag=1):
    """Echo dict for the JSON ``method_params`` field (provenance)."""
    return {
        "method": "symbolic_transfer_entropy",
        "embedding": {"m": m, "tau": tau},
        "symbolisation": "bandt_pompe_permutation",
        "alphabet_size": int(math.factorial(m)),
        "lag_symbol_steps": lag,
        "log_base": 2,
        "unit": "bits",
        "freq_band": "full",
        "fs": float(FS),
        "reference": "Schreiber 2000; Bandt & Pompe 2002",
    }
