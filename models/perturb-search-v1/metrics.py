"""metrics.py — 6 个输出端响应度量（design §4）+ null 地板（design §5 钉死构造）。

统一输入：同窗同 CRN 配对样本 samples (V, K, C, F)（行 0 = baseline，行 v ≥ 1 = 变体
(源 A, 范式 p, r)，行序 = candidates.variant_index）；δ_k = perturbed_k − baseline_k（靶通道
B 时间序列）。z-score 后数据，度量值无量纲。方向约定：行 = 源 A、列 = 靶 B。

null 地板（design §5 预注册）：baseline 行 K=16 拆 8+8 两半（h1 = base[:8] 作 pert 侧、
h2 = base[8:] 作 base 侧；归一化统计 s_{w,B} / env 分母取 base 侧 = h2），按各度量**同构**
计算两半间距离 → 每窗每靶通道一个 null 值（逐边 A→B 复用；z = D_edge / null_std_edge，
null_std = 逐边 24 个 null 值 std，见 scorecard.py）。

实现参数（config.py 钉死）：Welch fs=250/nperseg=128/Hann/50% overlap/bin 中心归属/
ε=1e-12 地板；M-env Hilbert 在 128 样本短段有边缘效应（pert/base/null 同算子施加，
偏差部分抵消，注记照登）；M-FC 逐 k 相关阵后均摊（128 时间点 > 62 通道）。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, sosfiltfilt, welch

import candidates as CD

_CHUNK = 65536        # welch/filtfilt/hilbert 逐块行数（内存护栏；不影响数值）
_CORR_CHUNK = 32      # M-FC 逐块变体行数


def _cfg():
    return CD.load_config()


# ---------------------------------------------------------------------------
# 特征提取（对任意 (..., F) 批量逐行分块；null 侧与变体侧共用同一函数 = 同构造）
# ---------------------------------------------------------------------------

def baseline_scale(base: np.ndarray) -> np.ndarray:
    """s_{w,B} = RMS_{k,t}(x0 − \\bar{k})（沿 response.py::baseline_scale 同构）：
    base (K, C, F) → (C,)；\\bar{k} = 逐 (B, t) 对 k 求均值。"""
    dev = base - base.mean(axis=0, keepdims=True)
    return np.sqrt((dev ** 2).mean(axis=(0, 2)))


def _psd(x: np.ndarray) -> np.ndarray:
    """Welch PSD（钉死参数）：x (..., F) → (..., nf)。detrend=scipy 默认 'constant'。"""
    SC = _cfg()
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = []
    for c0 in range(0, flat.shape[0], _CHUNK):
        _f, p = welch(flat[c0:c0 + _CHUNK], fs=SC.FS, window="hann",
                      nperseg=SC.WELCH_NPERSEG, noverlap=SC.WELCH_NOVERLAP, axis=-1)
        out.append(p)
    return np.concatenate(out, axis=0).reshape(shape[:-1] + (-1,))


def _welch_freqs() -> np.ndarray:
    SC = _cfg()
    _f, _p = welch(np.zeros(SC.WELCH_NPERSEG)[None, :], fs=SC.FS, window="hann",
                   nperseg=SC.WELCH_NPERSEG, noverlap=SC.WELCH_NOVERLAP, axis=-1)
    return _f


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """bin 中心落入带内为归属规则（闭区间）。"""
    return (freqs >= lo) & (freqs <= hi)


def _env_mean(x: np.ndarray) -> np.ndarray:
    """α 带 (8–13 Hz) Hilbert 包络时间均值：x (..., F) → (...,)。
    带通 = B 算子同款 Butterworth sos + sosfiltfilt 零相位；hilbert 逐行。"""
    SC = _cfg()
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = []
    for c0 in range(0, flat.shape[0], _CHUNK):
        bp = sosfiltfilt(CD.alpha_bandpass_sos(), flat[c0:c0 + _CHUNK], axis=-1)
        out.append(np.abs(hilbert(bp, axis=-1)).mean(axis=-1))
    return np.concatenate(out, axis=0).reshape(shape[:-1])


def _corr_mean(x: np.ndarray) -> np.ndarray:
    """M-FC 特征：逐 k 通道-通道 Pearson 相关阵后对 k 均摊：x (..., K, C, F) → (..., C, C)。
    r = Σxy / sqrt(Σx²·Σy²)（总体口径 ddof=0）；128 时间点 > 62 通道，估计量可行。"""
    lead = x.shape[:-3]
    K, C, F = x.shape[-3:]
    out = np.empty(lead + (C, C), dtype=x.dtype)
    flat = x.reshape(-1, K, C, F)
    n_rows = flat.shape[0]
    for c0 in range(0, n_rows, _CORR_CHUNK):
        c1 = min(c0 + _CORR_CHUNK, n_rows)
        blk = flat[c0:c1]
        blk = blk - blk.mean(axis=-1, keepdims=True)
        nrm = np.sqrt((blk ** 2).sum(axis=-1, keepdims=True))
        assert (nrm > 0).all(), "M-FC 零方差通道（合成数据请避开全零通道）"
        xn = blk / nrm
        R = np.einsum("qkcf,qkdf->qkcd", xn, xn)
        out.reshape(-1, C, C)[c0:c1] = R.mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# 距离函数（变体侧与 null 侧同构共用）
# ---------------------------------------------------------------------------

def d_rms(pert: np.ndarray, base: np.ndarray) -> np.ndarray:
    """M-RMS：RMS_{k,t}(δ)/s_{w,B}；pert/base (n, C, F) → (C,)。"""
    s = baseline_scale(base)
    return np.sqrt(((pert - base) ** 2).mean(axis=(0, 2))) / s


def d_psd(pert_L: np.ndarray, base_L: np.ndarray) -> np.ndarray:
    """M-PSD：mean_k ‖log10 PSD^pert − log10 PSD^base‖₂（1–40 Hz bin）；
    L = log10(max(psd, ε)) 已取过对数并选带，(n, C, nb) → (C,)。"""
    diff = pert_L - base_L
    return np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=0)


def d_band_primary(delta: np.ndarray) -> np.ndarray:
    """M-band 主标量：逐边取 |Δ| 最大带；delta (C, 5) = mean_k Δlog10 带功率 → (C,)。"""
    return np.abs(delta).max(axis=-1)


def d_env(pert_e: np.ndarray, base_e: np.ndarray) -> np.ndarray:
    """M-env：|mean_k env^pert − mean_k env^base| / mean_k env^base；e (n, C) → (C,)。"""
    return np.abs(pert_e.mean(axis=0) - base_e.mean(axis=0)) / base_e.mean(axis=0)


def d_fc(pert_R: np.ndarray, base_R: np.ndarray) -> np.ndarray:
    """M-FC：‖R_rowB^pert − R_rowB^base‖₂（均摊后相关阵行向量变化）；R (C, C) → (C,)。"""
    diff = pert_R - base_R
    return np.sqrt((diff ** 2).sum(axis=-1))


def d_time(pert: np.ndarray, base: np.ndarray):
    """M-time：δ̄(t) = mean_k δ_k(t)；主标量 = |δ̄| 前 32 样本均值；全段均值与峰值随附。
    pert/base (n, C, F) → (primary (C,), full (C,), peak (C,))。"""
    SC = _cfg()
    a = np.abs((pert - base).mean(axis=0))
    return (a[..., :SC.MTIME_EARLY_SAMPLES].mean(axis=-1),
            a.mean(axis=-1), a.max(axis=-1))


# ---------------------------------------------------------------------------
# 单窗全部度量计算（samples (V, K, C, F) float64 → {metric: 结果 dict}）
# ---------------------------------------------------------------------------

def compute_metrics(samples: np.ndarray, metrics) -> dict:
    """对一窗 CRN 批输出计算所请求度量。返回 {metric: {"D": (V-1, C), "null": (C,), ...}}；
    D 行序 v-1 = a·N_P + p（调用方 reshape (N_SRC, N_P, C) → transpose (N_P, A, B)）。
    M-band 附 "band_delta" (V-1, C, 5)；M-time 附 "mtime_full"/"mtime_peak" (V-1, C)。"""
    SC = _cfg()
    metrics = tuple(metrics)
    base = samples[0]
    n1, n2 = SC.NULL_SPLIT
    assert base.shape[0] == n1 + n2, f"baseline K={base.shape[0]} != NULL_SPLIT {n1}+{n2}"
    h1, h2 = base[:n1], base[n1:]
    out: dict = {}

    # ---- PSD 家族共享一次 Welch ----
    if "M-PSD" in metrics or "M-band" in metrics:
        freqs = _welch_freqs()
        eps = SC.PSD_EPS
        psd_all, psd_h1, psd_h2 = _psd(samples), _psd(h1), _psd(h2)
        band_bins = _band_mask(freqs, SC.PSD_BAND[0], SC.PSD_BAND[1])
        logp = np.log10(np.maximum(psd_all, eps))               # (V,K,C,nf)
        logp_h1 = np.log10(np.maximum(psd_h1, eps))
        logp_h2 = np.log10(np.maximum(psd_h2, eps))
        if "M-PSD" in metrics:
            L = logp[..., band_bins]
            D = np.sqrt(((L[1:] - L[0]) ** 2).sum(axis=-1)).mean(axis=1)   # (V-1,C)
            null = d_psd(logp_h1[..., band_bins], logp_h2[..., band_bins])
            out["M-PSD"] = {"D": D, "null": null}
        if "M-band" in metrics:
            band_idx = [np.where(_band_mask(freqs, lo, hi))[0] for _n, lo, hi in SC.BANDS]
            def _bp(pw):                                        # (...,nf)→(...,5) log10 带功率
                pb = np.stack([np.take(pw, idx, axis=-1).mean(axis=-1) for idx in band_idx], axis=-1)
                return np.log10(np.maximum(pb, eps))
            lpb, lpb1, lpb2 = _bp(psd_all), _bp(psd_h1), _bp(psd_h2)
            delta = (lpb[1:] - lpb[0]).mean(axis=1)             # (V-1,C,5) mean_k 先
            D = d_band_primary(delta)                           # 再逐边取 |Δ| 最大带
            null = d_band_primary((lpb1 - lpb2).mean(axis=0))
            out["M-band"] = {"D": D, "null": null, "band_delta": delta}

    if "M-RMS" in metrics:
        D = d_rms_rows(samples)
        null = d_rms(h1, h2)
        out["M-RMS"] = {"D": D, "null": null}

    if "M-env" in metrics:
        e = _env_mean(samples)                                   # (V,K,C)
        e0m = e[0].mean(axis=0)                                  # mean_k env^base (C,)
        D = np.abs(e[1:].mean(axis=1) - e0m) / e0m               # (V-1,C)
        null = d_env(_env_mean(h1), _env_mean(h2))
        out["M-env"] = {"D": D, "null": null}

    if "M-FC" in metrics:
        R = _corr_mean(samples)                                  # (V,C,C)
        D = d_fc(R[1:], np.broadcast_to(R[0], R[1:].shape))
        null = d_fc(_corr_mean(h1), _corr_mean(h2))
        out["M-FC"] = {"D": D, "null": null}

    if "M-time" in metrics:
        a = np.abs((samples[1:] - base).mean(axis=1))            # (V-1,C,F)
        D = a[..., :SC.MTIME_EARLY_SAMPLES].mean(axis=-1)
        null = d_time(h1, h2)[0]
        out["M-time"] = {"D": D, "null": null,
                         "mtime_full": a.mean(axis=-1), "mtime_peak": a.max(axis=-1)}
    for m in metrics:
        assert np.isfinite(out[m]["D"]).all() and np.isfinite(out[m]["null"]).all(), \
            f"{m} 出现非有限值"
    return out


def d_rms_rows(samples: np.ndarray) -> np.ndarray:
    """M-RMS 批量行版：samples (V, K, C, F) → (V-1, C)（逐变体行 vs baseline 行 0）。"""
    base = samples[0]
    s = baseline_scale(base)
    d = np.sqrt(((samples[1:] - base) ** 2).mean(axis=(1, 3))) / s
    return d
