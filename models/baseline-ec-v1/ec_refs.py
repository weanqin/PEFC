"""ec_refs —— 共享 VAR（单一全模型 Gram OLS + BIC）与两种观测参考读出。

- 条件 GC（主口径）：gc_psd.conditional_gc 的 Schur 读出，全模型条件化、行=源/列=靶、
  无逐边独立重拟合（满足 single-full-model 要求）。
- gPDC：与 GC 共享同一 Gram 拟合；readout=|gPDC|²（新息方差归一、源列归一，[0,1]、列平方和=1）；
  存储前显式转置为行=源（A_ij=目标 i 行/源 j 列 → [j,i]）。
- mne-connectivity GC：二元 state-space 交叉核对列（非主口径——逐连接建模、非全通道条件化，
  2026-10-13 实测确认；方向 seed→target 已实测验证）。

gc_psd 叶子函数只读复用（v1 代码零改动；numpy-only 口径不动）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

sys.path.insert(0, str(C.V1_DIR))
from gc_psd import bic_from_gram, conditional_gc, gram_pieces, var_design  # noqa: E402


# ---------- Gram / BIC ----------

def fit_gram(x: np.ndarray, p: int) -> Dict:
    """整段单窗 Gram（x: (C,T) 连续记录；row_from=p 正常行集）。"""
    assert x.ndim == 2
    return gram_pieces(x[None].astype(np.float64), p, row_from=p)


def bic_pass(x: np.ndarray, p_max: int = C.P_MAX) -> Dict:
    """BIC 选阶（对齐行集 row_from=p_max 的嵌套口径）。C=62 时惩罚重，仅作存档参考。"""
    gram = gram_pieces(x[None].astype(np.float64), p_max, row_from=p_max)
    p_star, table = bic_from_gram(gram)
    return {"p_star": int(p_star), "table": table}


# ---------- ridge 拟合（2026-10-13 定标：OLS 在 62ch/(85-170)s 下条件数 ~1e18、
# split-half 天花板 ≈0，Schur 读出被共线噪声支配——见 experiment.md Run 001 证据；
# λ_rel=1e-2 由三被试 split-half 稳定性校准（0.91-0.94），GCV 欠平滑弃用；p=8 按
# ASCT「p/2 谱峰 × EEG 2-4 节律」锚点固定） ----------

P_SHARED = 8          # 固定共用阶（125 Hz；8 lags = 64 ms）
LAM_REL = 1e-2        # 相对惩罚（λ = LAM_REL × tr(G_lag)/k_lag；截距不罚）

WHITENESS_GRID = (4, 8, 12, 16, 20, 24)   # 白性阶梯：仅存档 QC（LB 对真实数据超严，
WHITENESS_THRESH = 0.25                    # lb_frac≈1 属常态，不作选阶判据）


def ridge_fit(gram: Dict, lam_rel: float = LAM_REL) -> Dict:
    """ridge 全模型：B = (G+diag(0,λ…))⁻¹ Zy；SSR 二次型 = YtY−2BᵀZy+BᵀGB；Σ=SSR/n。"""
    G, Zy = gram["G"], gram["Zy"]
    k = G.shape[0]
    lam = lam_rel * float(np.trace(G[1:, 1:])) / (k - 1)
    P = np.zeros(k); P[1:] = lam                    # 截距不罚
    A = G + np.diag(P)
    B = np.linalg.solve(A, Zy)
    SSR = gram["YtY"] - 2.0 * B.T @ Zy + B.T @ G @ B
    Sigma = (SSR + SSR.T) / 2.0 / gram["n"]
    return {"B": B, "Sigma": Sigma, "lam": lam, "ssr_diag": np.diag(SSR).copy(),
            "n": gram["n"]}


def ridge_conditional_gc(gram: Dict, lam_rel: float = LAM_REL) -> np.ndarray:
    """ridge 条件 GC：对每个源 j 删其滞后列、同一 λ 重解（62 次，非 3782 次）。

    gc[源, 靶] = ln(SSR_red_j[靶]/SSR_full[靶])；ridge 下可为微负（收缩口径，登记），
    排序语义不受影响。
    """
    G, Zy, YtY = gram["G"], gram["Zy"], gram["YtY"]
    C = Zy.shape[1]
    k = G.shape[0]
    p = (k - 1) // C
    lam = lam_rel * float(np.trace(G[1:, 1:])) / (k - 1)
    full = ridge_fit(gram, lam_rel)
    cols = np.arange(1, k)
    src_of_col = (cols - 1) % C
    gc = np.zeros((C, C))
    for j in range(C):
        keep = np.concatenate([[0], cols[src_of_col != j]])
        Gr = G[np.ix_(keep, keep)]
        Zr = Zy[keep]
        Pr = np.zeros(len(keep)); Pr[1:] = lam
        Br = np.linalg.solve(Gr + np.diag(Pr), Zr)
        ssr_red = np.diag(YtY - 2.0 * Br.T @ Zr + Br.T @ Gr @ Br)
        ii = np.arange(C) != j
        gc[j, ii] = np.log(np.maximum(ssr_red[ii] / full["ssr_diag"][ii], 1e-12))
    return gc


def choose_p_whiteness(x: np.ndarray, p_max: int = C.P_MAX,
                       grid=WHITENESS_GRID, thresh: float = WHITENESS_THRESH) -> Dict:
    """白性阶梯（仅存档 QC 用；选阶已改固定 p=8，见模块头注释）。"""
    rows = []
    for p in grid:
        gram = gram_pieces(x[None].astype(np.float64), p, row_from=p)
        co = ridge_fit(gram)
        rows.append({"p": p, "lb_frac": round(ljung_box_frac(x, p, co["B"]), 4)})
    return {"ladder": rows}


def choose_p_shared(p_stars: List[int]) -> int:
    """全体共用阶：中位数（跨被试可比性优先；分布分歧登记于 summary）。"""
    return int(np.median(p_stars))


# ---------- 系数与新息协方差（gPDC 与残差 QC 共用） ----------

def fit_coeffs(gram: Dict, p: int) -> Dict:
    """B (1+Cp, C)：行=预测子列 [截距, lag1 全通道, ...]，列=目标；Σ = SSR/n（ML 口径）。"""
    B = np.linalg.solve(gram["G"], gram["Zy"])
    SSR = gram["YtY"] - B.T @ gram["Zy"]
    Sig = (SSR + SSR.T) / 2.0 / gram["n"]
    return {"B": B, "Sigma": Sig, "n": gram["n"]}


def residuals(x: np.ndarray, p: int, B: np.ndarray) -> np.ndarray:
    Z, Y = var_design(x[None].astype(np.float64), p)
    return Y - Z @ B


def ljung_box_frac(x: np.ndarray, p: int, B: np.ndarray, K: Optional[int] = None) -> float:
    """残差白性 QC：各通道 Ljung-Box(滞后 1..K, 默认 2p) p<0.01 的通道占比。"""
    from scipy.stats import chi2
    e = residuals(x, p, B)                       # (n, C)
    n = e.shape[0]
    K = K or 2 * p
    ec = e - e.mean(0, keepdims=True)
    denom = (ec ** 2).sum(0)
    Q = np.zeros(e.shape[1])
    for k in range(1, K + 1):
        rk = (ec[k:] * ec[:-k]).sum(0) / np.maximum(denom, 1e-300)
        Q += rk ** 2 / (n - k)
    Q *= n * (n + 2)
    pv = chi2.sf(Q, K)
    return float((pv < 0.01).mean())


# ---------- gPDC（|gPDC|²，行=源存储） ----------

def gpdc_squared(B: np.ndarray, Sigma: np.ndarray, fs: float = C.FS,
                 n_fft: int = C.N_FFT) -> Dict:
    """返回 {freqs, g2 (C,C,F) 行=源/列=靶, broadband (C,C), bands dict}。

    Ā_ij(f) = Σ_l (A_l[i,j]/σ_i) e^{-i2πf l/fs}（i=目标行, j=源列）；
    gPDC²_ij(f) = |Ā_ij(f)|² / Σ_k |Ā_kj(f)|²（源列 j 归一）→ [0,1]、列平方和=1；
    存储转置 [j,i]（行=源）。
    """
    n_pred, Cc = B.shape
    p = (n_pred - 1) // Cc
    A = np.stack([B[1 + (l - 1) * Cc: 1 + l * Cc, :].T for l in range(1, p + 1)],
                 axis=-1)                                    # (C_tgt, C_src, p)
    sig = np.sqrt(np.maximum(np.diag(Sigma), 1e-300))
    Abar = A / sig[:, None, None]
    Af = np.fft.rfft(Abar, n=n_fft, axis=-1)                 # (C_tgt, C_src, F)
    mag2 = np.abs(Af) ** 2
    denom = mag2.sum(axis=0, keepdims=True)                  # Σ_k over 目标行
    denom = np.maximum(denom, 1e-300)
    g2 = (mag2 / denom).transpose(1, 0, 2)                   # → (C_src, C_tgt, F) 行=源
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    out = {"freqs": freqs, "g2": g2}
    for name, band in {**C.BANDS, "broadband": C.BROADBAND}.items():
        m = (freqs >= band[0]) & (freqs < band[1] + 1e-9)
        out[name] = g2[..., m].mean(-1)
    return out


# ---------- 条件 GC（主口径，ridge 读出） ----------

def conditional_gc_matrix(gram: Dict, p: Optional[int] = None,
                          lam_rel: float = LAM_REL) -> np.ndarray:
    """gc[源, 靶] = ln(SSR_red/SSR_full)（行=源；全模型条件化；ridge 收缩口径）。"""
    if p is not None:
        Cc = gram["Zy"].shape[1]
        assert (gram["G"].shape[0] - 1) // Cc == p, "gram 阶数与传入 p 不一致"
    return ridge_conditional_gc(gram, lam_rel=lam_rel)


# ---------- mne-connectivity 二元交叉核对（非主口径） ----------

def mnec_gc_matrix(x: np.ndarray, p: int, fs: float = C.FS,
                   fdecim: int = C.MNEC_FDECIM) -> np.ndarray:
    """二元 state-space GC（seed→target 方向），1–40 Hz 带平均，返回 行=源 矩阵。"""
    from mne_connectivity import spectral_connectivity_epochs
    Cc = x.shape[0]
    pairs = [(i, j) for i in range(Cc) for j in range(Cc) if i != j]
    indices = ([[s] for s, _ in pairs], [[t] for _, t in pairs])
    con = spectral_connectivity_epochs(
        x[None].astype(np.float64), method="gc", mode="multitaper",
        indices=indices, sfreq=fs, fmin=C.BROADBAND[0], fmax=C.BROADBAND[1],
        gc_n_lags=int(p), fdecim=fdecim, n_jobs=1, verbose="ERROR")
    f = np.asarray(con.freqs)
    vals = np.asarray(con.get_data())[:, (f >= C.BROADBAND[0]) & (f <= C.BROADBAND[1])].mean(-1)
    M = np.zeros((Cc, Cc))
    for (s, t), v in zip(pairs, vals):
        M[s, t] = v
    return M


# ---------- 单被试全流程 ----------

def refs_for_subject(x: np.ndarray, p: int = P_SHARED,
                     mnec: bool = False) -> Dict:
    """固定阶 + ridge → {gc, gpdc_bb, gpdc_bands, (mnec), lb_frac, meta}。"""
    gram = fit_gram(x, p)
    gc = conditional_gc_matrix(gram, lam_rel=LAM_REL)
    co = ridge_fit(gram, lam_rel=LAM_REL)
    gp = gpdc_squared(co["B"], co["Sigma"])
    out = {
        "gc": gc,
        "gpdc_bb": gp["broadband"],
        "gpdc_bands": np.stack([gp[k] for k in C.BANDS]),
        "lb_frac": ljung_box_frac(x, p, co["B"]),
        "n": gram["n"], "p": p, "lam": co["lam"],
    }
    if mnec:
        out["mnec"] = mnec_gc_matrix(x, p)
    return out
