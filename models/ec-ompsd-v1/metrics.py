"""metrics.py — M-PSD 度量（10-18-ec-ompsd-f0；口径同 perturb-search-v1 钉死值）。

输入：同窗同 CRN 配对样本 samples (V, K, C, F)（行 0 = baseline）。
输出：{"D": (V−1, C), "logpsd": (V, K, C, nb)}——D = mean_k ‖log10 PSD^pert −
log10 PSD^base‖₂（1–40 Hz bin 中心归属）；Welch nperseg=128/Hann/50% overlap/
detrend='constant'/ε=1e-12 地板。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

import candidates as CD

_CHUNK = 65536


def _psd(x: np.ndarray) -> np.ndarray:
    SC = CD.load_config()
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = []
    for c0 in range(0, flat.shape[0], _CHUNK):
        _f, p = welch(flat[c0:c0 + _CHUNK], fs=SC.FS, window="hann",
                      nperseg=SC.WELCH_NPERSEG, noverlap=SC.WELCH_NOVERLAP, axis=-1)
        out.append(p)
    return np.concatenate(out, axis=0).reshape(shape[:-1] + (-1,))


def compute(samples: np.ndarray) -> dict:
    """M-PSD 主标量（D）+ 全谱留档（logpsd，含 baseline 行）。"""
    SC = CD.load_config()
    _f, _p = welch(np.zeros(SC.WELCH_NPERSEG)[None, :], fs=SC.FS, window="hann",
                   nperseg=SC.WELCH_NPERSEG, noverlap=SC.WELCH_NOVERLAP, axis=-1)
    band = (_f >= SC.PSD_BAND[0]) & (_f <= SC.PSD_BAND[1])
    logp = np.log10(np.maximum(_psd(np.asarray(samples, dtype=np.float64)), SC.PSD_EPS))
    L = logp[..., band]                                     # (V,K,C,nb)
    D = np.sqrt(((L[1:] - L[0]) ** 2).sum(axis=-1)).mean(axis=1)   # (V−1,C)
    return {"D": D, "logpsd": logp}
