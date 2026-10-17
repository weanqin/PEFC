#!/usr/bin/env python3
"""psd_band_metric3.py — 指标 3 预登记口径补算（design §1.4-4；数据源 eval/psd.npz，只读）。

口径（2026-10-16 预登记，evaluation-calibers「指标定义先于运行」）：
- 通过判据计算式 = 组平均谱（全通道平均）在 **1–30 Hz** 的 mean |Δlog10| ≤ 0.5
  （下沿 1 Hz 对齐 release 带通——真实侧 <1 Hz 为滤波地板，不得入口径窗）
- 高频观察段 **40–125 Hz**（Nyquist 125 Hz；无阈值，照登数值）
- >40 Hz 功率占比 = Σ_{f>40} PSD / Σ_{f>0} PSD（D3 观察项：无低通口径下生成侧是否自行待在带内）
- 全频段 mean_abs_logdiff（summary.json 现有统计量，自然对数）在新口径下不作判读依据
  （生成侧未滤波 + 真实侧 1–40 Hz 地板，被地板区主导）——仅照登对照

用法：python3 psd_band_metric3.py <run_dir> [<run_dir> ...]（run_dir 含 eval/psd.npz）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def band_logdiff(freqs: np.ndarray, spec_r: np.ndarray, spec_g: np.ndarray,
                 lo: float, hi: float) -> float:
    """组平均谱在 [lo, hi] Hz 的 mean |Δlog10|（1e-30 数值地板防 log(0)）。"""
    m = (freqs >= lo) & (freqs <= hi)
    lr = np.log10(np.maximum(spec_r[m], 1e-30))
    lg = np.log10(np.maximum(spec_g[m], 1e-30))
    return float(np.abs(lr - lg).mean())


def hf_power_fraction(freqs: np.ndarray, spec: np.ndarray, cut: float = 40.0) -> float:
    """>40 Hz 功率占比 = Σ_{f>cut} / Σ_{f>0}（组平均谱）。"""
    pos = freqs > 0
    hi = freqs > cut
    return float(spec[hi].sum() / max(spec[pos].sum(), 1e-30))


def main() -> None:
    for rd in sys.argv[1:]:
        rd = Path(rd)
        d = np.load(rd / "eval" / "psd.npz")
        freqs, psd_r, psd_g = d["freqs"], d["psd_real"], d["psd_gen"]
        mr, mg = psd_r.mean(0), psd_g.mean(0)  # 组平均谱（全通道平均）
        pos = freqs > 0
        out = {
            "run": str(rd),
            "n_segments": int(d["n_segments"]),
            "nyquist_hz": float(freqs[-1]),
            "band_1_30_mean_abs_dlog10": round(band_logdiff(freqs, mr, mg, 1.0, 30.0), 5),
            "pass_1_30_le_0.5": bool(band_logdiff(freqs, mr, mg, 1.0, 30.0) <= 0.5),
            "band_40_125_mean_abs_dlog10": round(band_logdiff(freqs, mr, mg, 40.0, 125.0), 5),
            "band_0_1_mean_abs_dlog10": round(band_logdiff(freqs, mr, mg, 0.0, 1.0), 5),
            "hf_power_frac_gt40_real": round(hf_power_fraction(freqs, mr), 6),
            "hf_power_frac_gt40_gen": round(hf_power_fraction(freqs, mg), 6),
            "peak_hz_real": float(freqs[pos][mr[pos].argmax()]),
            "peak_hz_gen": float(freqs[pos][mg[pos].argmax()]),
            "fullband_mean_abs_logdiff_ln": round(float(np.abs(
                np.log(np.maximum(mr[pos], 1e-30)) - np.log(np.maximum(mg[pos], 1e-30))).mean()), 5),
        }
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
