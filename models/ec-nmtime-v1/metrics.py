"""metrics.py — M-time 度量（10-18-ec-nmtime-f0；口径同 perturb-search-f0 钉死值）。

输入：同窗同 CRN 配对样本 samples (V, K, C, F)（行 0 = baseline，行 v ≥ 1 = 变体
(源 A, 格点 j)）；δ_k = perturbed_k − baseline_k。输出逐行 D (V−1, C)。
"""

from __future__ import annotations

import numpy as np

import candidates as CD


def d_time(samples: np.ndarray) -> dict:
    """M-time：δ̄(t) = mean_k δ_k(t) → a(t) = |δ̄(t)| → D = 前 32 样本均值（主标量）。

    随附登记（不入 readout）：mtime_full = 全段均值、mtime_peak = 全段峰值。
    samples (V, K, C, F) → {"D": (V−1, C), "mtime_full": (V−1, C), "mtime_peak": (V−1, C)}。
    """
    SC = CD.load_config()
    base = samples[0]
    a = np.abs((samples[1:] - base).mean(axis=1))            # (V−1, C, F) CRN 集群均值绝对值
    return {"D": a[..., :SC.MTIME_EARLY_SAMPLES].mean(axis=-1),
            "mtime_full": a.mean(axis=-1), "mtime_peak": a.max(axis=-1)}
