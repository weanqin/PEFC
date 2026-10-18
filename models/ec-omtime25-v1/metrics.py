"""metrics.py — M-time@25 度量（10-18-ec-omtime25-f0；主标量 = |δ̄| 前 25 样本 = 0.1 s）。

输入：同窗同 CRN 配对样本 (V, K, C, F)，行 0 = baseline；δ_k = pert_k − base_k。
输出：{"D": (V−1, C), "mtime_full": …, "mtime_peak": …}（full/peak 随附登记，不入 readout）。
"""

from __future__ import annotations

import numpy as np

import candidates as CD


def compute(samples: np.ndarray) -> dict:
    SC = CD.load_config()
    base = samples[0]
    a = np.abs((samples[1:] - base).mean(axis=1))            # (V−1, C, F) |δ̄(t)|
    return {"D": a[..., :SC.MTIME_EARLY_SAMPLES].mean(axis=-1),
            "mtime_full": a.mean(axis=-1), "mtime_peak": a.max(axis=-1)}
