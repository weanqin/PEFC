#!/usr/bin/env python3
"""gc_psd.py — GC / PSD / 空间结构归因（design §6 指标 1/3/4；implement.md 6.3）。

numpy-only（无 scipy/statsmodels——全部手写：Welch、BH-FDR、Spearman=秩变换 Pearson、
F/t 分布 p 值 = 正则化不完全 Beta 连分式）。

GC（future-only 口径，两组参数完全一致）：
- 全模型 VAR(p) OLS：堆叠测试折 future 128 点段（真实侧 = 同位置真实 future；
  生成侧 = 每窗固定 seed 1 条样本，evaluate.py 已落盘 eval/gc_input.npz）；
  Gram 矩阵（ZᵀZ / ZᵀY / YᵀY）按受试者分块算出后聚合（避免整折设计矩阵 GB 级物化）；
- **BIC 在真实组上选 p∈{1..p_max} 后两组共用**（对齐行集 t∈[p_max,F) 保证嵌套子矩阵口径，
  登记）；BIC = ln det Σ̂ + C(1+Cp)·ln(n)/n；
- 成对条件 GC：列删除的 OLS 额外平方和公式（块 Schur 补）——SSR_r[i,i] = SSR_full[i,i]
  + B[cols_j,i]ᵀ·(G⁻¹)[cols_j,cols_j]⁻¹·B[cols_j,i]，全组一次求逆 + 62×(p×p) 小逆，
  **不做 3782 次全量重拟合**；GC 量 = ln(SSR_r/SSR_full)，F 统计 = 经典联合显著性 F 检验；
- 62×62 有向矩阵去对角展平：两组 Pearson r + Spearman ρ + 95% CI + raw p；
  BH-FDR（numpy 排序实现）对各组 3782 条边 raw p 校正（q<0.05 显著边数与两组 Jaccard）；
  bootstrap：**受试者级**（10 测试受试者 Gram 重采样，两组配对同索引，1000 次登记；
  窗级 Gram 存储不可行——逐窗 621×621 f32 ≈ 20 GB，且 stride=64 相邻窗高度相关，
  受试者才是独立单元——此为对「对窗 bootstrap」的可实现偏离，登记）。

PSD（两组完全相同分段与拼接）：测试折抽取互不重叠 future 段（窗 start ≡ 0 mod 2·stride →
future 首尾相接），逐受试者拼成连续序列，丢前导残段对齐绝对 512 点网格后切 512 点块 →
Welch（Hann，nperseg=512，分辨率 = fs/nperseg——fs=500 → 0.977 Hz、fs=250 → 0.488 Hz；
fs 由调用方传入，250 Hz fmt 数据集经 evaluate.py 传 cfg.fs）。

skip_gc default-off 钩子（用户 2026-10-16 拍板 D1：genGC vs real GC 对比暂时阻塞）：
run_all(skip_gc=True) 跳过阶段 1/2（BIC 选阶 + 两组 VAR 条件 GC + 一致性/bootstrap），
**PSD 与空间结构归因照跑**（二者只依赖 gc_input.npz 的 gen/real/win 元数据，与 GC 阶段
无数据依赖）；summary/日志显式记 "gc": "skipped(blocked)"，且不落任何 GC 产物（gc_real/
gc_gen/gc_F_p_* npy 与 gc 矩阵图、结果 json 的 gc 数组/图引用）。默认 False = 原行为逐位不变。

指标 4（空间结构归因）：生成样本 62×62 零滞后相关 R̂ vs K_s（Ricker3D σ=0.080，加 jitter 前
原始核）vs 真实 R 三方对比（Frobenius + 上三角 Pearson/Spearman + 数值表）。

各阶段打印进度与 ETA（R9）；结果落 eval/（json+npy）与 figs/（经 common/visualizer）。
可独立运行：python3 models/gpfm-mamba-v1/gc_psd.py --run-id main-01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # 仓库根 -> common/

from gpr_prior import ricker3d_kernel  # noqa: E402

BOOTSTRAP_SEED = 2026  # 登记：GC 相关 CI bootstrap 种子（受试者级，与采样 seed 独立）
GRAM_SEG_CHUNK = 100   # Gram 分块段数（控设计矩阵瞬时内存）


# ================= 概率分布（numpy-only 手写） =================

def _betacf(a: float, b: float, x: float, max_iter: int = 600, eps: float = 3e-12) -> float:
    """正则化不完全 Beta 的连分式（Modified Lentz）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lfront = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lfront) * _betacf(a, b, x) / a
    lcomplement = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                   + b * math.log1p(-x) + a * math.log(x))
    return 1.0 - math.exp(lcomplement) * _betacf(b, a, 1.0 - x) / b


def f_sf(x: float, d1: float, d2: float) -> float:
    """F(d1,d2) 分布生存函数：sf = I_{d2/(d2+d1 x)}(d2/2, d1/2)。"""
    if x <= 0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))


def t_two_sided_p(t: float, nu: float) -> float:
    """双侧 t 检验 p：p = I_{nu/(nu+t^2)}(nu/2, 1/2)。"""
    if t == 0:
        return 1.0
    return betainc(nu / 2.0, 0.5, nu / (nu + t * t))


# ================= 基础统计 =================

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xz = x - x.mean()
    yz = y - y.mean()
    denom = math.sqrt(float((xz * xz).sum()) * float((yz * yz).sum()))
    if denom == 0.0:
        return 0.0
    return float(np.clip((xz * yz).sum() / denom, -1.0, 1.0))


def rank_transform(x: np.ndarray) -> np.ndarray:
    """序数秩（升序 1..n；无并列校正——连续数据口径，登记）。"""
    order = np.argsort(np.asarray(x), kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_r(rank_transform(x), rank_transform(y))


def r_and_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson r + 朴素 t 检验 raw p（n = 展平元素数；GC 矩阵元素非独立——p 为朴素口径，登记）。"""
    r = pearson_r(x, y)
    n = len(x)
    if n <= 2:
        return r, 1.0
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t_two_sided_p(t, n - 2)


def bh_fdr(p_raw: np.ndarray) -> np.ndarray:
    """BH-FDR 校正 q 值（numpy 排序实现：p_sorted·m/rank 取后缀最小，截 [0,1]）。"""
    p = np.asarray(p_raw, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


# ================= VAR / 条件 GC =================

def var_design(X: np.ndarray, p: int, row_from: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """窗堆叠 VAR 设计（X: (N_seg, C, F)；row_from=None 时取 p）。

    行 = 各段 t∈[row_from, F)；Z 列 = [截距, lag1 全通道, ..., lag_p 全通道]，Y = X[:, t]。
    列序保证 p' < p 的模型是列子集（BIC 嵌套口径，配同一 row_from 使用）。
    """
    N, C, F = X.shape
    r0 = p if row_from is None else row_from
    R = F - r0
    Z = np.ones((N * R, 1 + C * p), dtype=np.float64)
    for l in range(1, p + 1):
        Z[:, 1 + (l - 1) * C : 1 + l * C] = X[:, :, r0 - l : F - l].transpose(0, 2, 1).reshape(N * R, C)
    Y = X[:, :, r0:].transpose(0, 2, 1).reshape(N * R, C)
    return Z, Y


def gram_pieces(X: np.ndarray, p: int, row_from: Optional[int] = None) -> Dict:
    """Gram 三件套 ZᵀZ / ZᵀY / YᵀY（+ 行数），float64，按段分块累积控内存。"""
    acc = None
    for c0 in range(0, X.shape[0], GRAM_SEG_CHUNK):
        Z, Y = var_design(X[c0 : c0 + GRAM_SEG_CHUNK], p, row_from)
        g = {"G": Z.T @ Z, "Zy": Z.T @ Y, "YtY": Y.T @ Y, "n": float(Z.shape[0])}
        acc = g if acc is None else {k: acc[k] + g[k] for k in g}
    return acc


def sum_grams(pieces: Sequence[Dict]) -> Dict:
    return {
        "G": np.sum([q["G"] for q in pieces], axis=0),
        "Zy": np.sum([q["Zy"] for q in pieces], axis=0),
        "YtY": np.sum([q["YtY"] for q in pieces], axis=0),
        "n": float(np.sum([q["n"] for q in pieces])),
    }


def _sub_gram(gram: Dict, p: int) -> Dict:
    """p' < p 的嵌套子 Gram（须同一对齐行集构建）。"""
    C = gram["Zy"].shape[1]
    k = 1 + C * p
    return {"G": gram["G"][:k, :k], "Zy": gram["Zy"][:k, :], "YtY": gram["YtY"], "n": gram["n"]}


def bic_from_gram(gram_max: Dict, log: Optional[Callable] = None) -> Tuple[int, List[dict]]:
    """真实组对齐行集 Gram 上 BIC 选 p∈{1..p_max}（嵌套子矩阵）。

    BIC = ln det Σ̂_p + C(1+Cp)·ln(n)/n（Σ̂ = SSR/n，ML 口径；logdet 用 Cholesky）。
    """
    C = gram_max["Zy"].shape[1]
    p_max = (gram_max["G"].shape[0] - 1) // C
    n = gram_max["n"]
    tab = []
    for p in range(1, p_max + 1):
        g = _sub_gram(gram_max, p)
        try:
            B = np.linalg.solve(g["G"], g["Zy"])
            SSR = g["YtY"] - B.T @ g["Zy"]
            Sig = (SSR + SSR.T) / 2 / g["n"]
            Sig = Sig + np.eye(C) * (1e-12 * max(float(np.trace(Sig)) / C, 1e-300))
            L = np.linalg.cholesky(Sig)
            logdet = 2.0 * float(np.log(np.diag(L)).sum())
            bic = logdet + C * (1 + C * p) * math.log(n) / n
        except np.linalg.LinAlgError:
            bic, logdet = math.inf, math.nan
        tab.append({"p": p, "logdet": None if math.isnan(logdet) else round(logdet, 4),
                    "bic": None if math.isinf(bic) else round(bic, 4), "n_rows": int(n)})
        if log:
            log(f"[gc-bic] p={p:2d} logdet={logdet:.4f} BIC={bic:.4f}")
    best = min(tab, key=lambda r: math.inf if r["bic"] is None else r["bic"])["p"]
    if log:
        log(f"[gc-bic] 真实组选定 p*={best}（两组共用，登记；对齐行集 t∈[{p_max},F)）")
    return best, tab


def bic_select_p(X: np.ndarray, p_max: int, log: Optional[Callable] = None) -> Tuple[int, List[dict]]:
    """小数据便捷封装（单测用；正式流程走按受试者 Gram + bic_from_gram）。"""
    return bic_from_gram(gram_pieces(X, p_max, row_from=p_max), log=log)


def conditional_gc(gram: Dict, p: int, with_tests: bool = True) -> Dict:
    """成对条件 GC（列删除 OLS 额外平方和 / 块 Schur 补，不做 3782 次全量重拟合）。

    对每个源通道 j：删 j 的 p 个滞后列 → 响应 i 的
    SSR_r[i,i] = SSR_full[i,i] + Δ_ij，Δ_ij = B[cols_j,i]ᵀ·(G⁻¹)[cols_j,cols_j]⁻¹·B[cols_j,i]
    （OLS 删列的额外平方和公式；删响应列不影响其余方程的 OLS）。
    返回 gc（ln SSR 比，行=源 j 列=靶 i）、F 统计、raw p（F(p, n−k)）。
    with_tests=False 跳过逐边 p 值（bootstrap 内层提速；betainc 逐边较贵）。
    """
    Sy = gram["Zy"]
    C = Sy.shape[1]
    n = gram["n"]
    k = 1 + C * p
    df2 = n - k
    if df2 <= 0:
        raise ValueError(f"行数不足：n={n}, k={k}")
    B = np.linalg.solve(gram["G"], Sy)
    SSR = gram["YtY"] - B.T @ Sy
    SSRd = np.diag(SSR).copy()
    if np.any(SSRd <= 0):
        bad = int(np.sum(SSRd <= 0))
        SSRd = np.maximum(SSRd, 1e-12 * max(float(SSRd.max()), 1e-300))
        print(f"[gc] 警告：{bad} 个响应 SSR_full ≤ 0（数值下限护栏生效）", flush=True)
    Ginv = np.linalg.inv(gram["G"])
    gc = np.zeros((C, C))
    Fm = np.zeros((C, C))
    Pm = np.ones((C, C))
    for j in range(C):
        cols = 1 + np.arange(p) * C + j
        S = Ginv[np.ix_(cols, cols)]
        bmat = B[cols]                                     # (p, C)
        delta = np.einsum("pi,pq,qi->i", bmat, np.linalg.inv(S), bmat)
        ratio = (SSRd + delta) / SSRd
        idx_ = np.arange(C) != j
        gc[j, idx_] = np.log(np.maximum(ratio[idx_], 1e-300))
        if with_tests:
            Fj = (delta / p) / (SSRd / df2)
            Fm[j, idx_] = Fj[idx_]
            for i in np.where(idx_)[0]:
                Pm[j, i] = f_sf(Fj[i], p, df2)
    return {"gc": gc, "F": Fm, "p": Pm, "n": n, "p_order": p}


# ================= Welch PSD（numpy-only） =================

def welch_psd(x: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch（Hann、逐段去均值、单边功率谱密度 V²/Hz）。x 为 1D（自动切 nperseg 段丢尾）。

    返回 (freqs, psd)，psd = 各段 |rfft(x·w)|² 平均 / (fs·Σw²)，非 DC 频率 ×2。
    """
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1, "welch_psd 单通道 1D 输入（多通道逐通道调用）"
    m = len(x) // nperseg
    if m == 0:
        raise ValueError(f"长度 {len(x)} < nperseg {nperseg}")
    segs = x[: m * nperseg].reshape(m, nperseg)
    win = np.hanning(nperseg)
    xw = (segs - segs.mean(axis=1, keepdims=True)) * win
    Xf = np.fft.rfft(xw, axis=1)
    psd = (np.abs(Xf) ** 2).mean(axis=0) / (fs * float((win ** 2).sum()))
    psd[1:] *= 2.0
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


# ================= 零滞后空间相关 =================

def zero_lag_corr(X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """X: (N, C, F) → C×C 零滞后相关（分块累积协方差，float64；常数通道相关置 0）。"""
    Xf = np.asarray(X, dtype=np.float64)
    C = Xf.shape[1]
    flat = Xf.transpose(1, 0, 2).reshape(C, -1)            # (C, N*F)
    mu = flat.mean(axis=1)
    sd = flat.std(axis=1)
    sd = np.where(sd < 1e-12, np.inf, sd)
    cov = np.zeros((C, C))
    total = flat.shape[1]
    for c0 in range(0, total, chunk):
        blk = (flat[:, c0 : c0 + chunk] - mu[:, None]) / sd[:, None]
        cov += blk @ blk.T
    corr = np.clip(cov / total, -1, 1)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def tri_flat(M: np.ndarray) -> np.ndarray:
    """上三角（去对角）展平——对称矩阵（空间相关三方对比）用。"""
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def offdiag_flat(M: np.ndarray) -> np.ndarray:
    """去对角全展平——有向矩阵（GC，62×61=3782 条有向边）用。"""
    return M[~np.eye(M.shape[0], dtype=bool)]


def three_way(R_hat: np.ndarray, K_s: np.ndarray, R_real: np.ndarray) -> Dict:
    out = {
        "fro_Rhat_Ks": round(float(np.linalg.norm(R_hat - K_s)), 5),
        "fro_Rhat_R": round(float(np.linalg.norm(R_hat - R_real)), 5),
        "fro_Ks_R": round(float(np.linalg.norm(K_s - R_real)), 5),
    }
    for name, (A, B) in {
        "tri_r_Rhat_Ks": (R_hat, K_s), "tri_r_Rhat_R": (R_hat, R_real),
        "tri_r_Ks_R": (K_s, R_real),
    }.items():
        out[name] = round(pearson_r(tri_flat(A), tri_flat(B)), 5)
        out[name.replace("_r_", "_rho_")] = round(spearman_rho(tri_flat(A), tri_flat(B)), 5)
    return out


# ================= 主流程 =================

def _bootstrap_corr_ci(pieces_real: List[Dict], pieces_gen: List[Dict], p: int,
                       n_boot: int, seed: int, log: Callable) -> Dict:
    """受试者级配对 bootstrap（两组同重采样索引）→ GC 矩阵 Pearson/Spearman 的 95% CI。"""
    rng = np.random.default_rng(seed)
    S = len(pieces_real)
    rs, rhos = [], []
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, S, S)
        mr = conditional_gc(sum_grams([pieces_real[i] for i in idx]), p, with_tests=False)["gc"]
        mg = conditional_gc(sum_grams([pieces_gen[i] for i in idx]), p, with_tests=False)["gc"]
        xr, yg = offdiag_flat(mr), offdiag_flat(mg)
        rs.append(pearson_r(xr, yg))
        rhos.append(spearman_rho(xr, yg))
        if (b + 1) % 100 == 0 or b == n_boot - 1:
            el = time.time() - t0
            log(f"[gc-boot] {b + 1}/{n_boot} | {el:.0f}s | ETA {el / (b + 1) * (n_boot - b - 1):.0f}s")
    rs, rhos = np.array(rs), np.array(rhos)
    return {
        "bootstrap_unit": "subject", "n_boot": n_boot, "seed": seed,
        "pearson_ci95": [round(float(np.percentile(rs, 2.5)), 5),
                          round(float(np.percentile(rs, 97.5)), 5)],
        "spearman_ci95": [round(float(np.percentile(rhos, 2.5)), 5),
                          round(float(np.percentile(rhos, 97.5)), 5)],
        "pearson_boot_mean": round(float(rs.mean()), 5),
        "spearman_boot_mean": round(float(rhos.mean()), 5),
        "note": "受试者级重采样（窗级 Gram 存储不可行且相邻窗相关；对「对窗 bootstrap」的登记偏离）",
    }


def run_all(run_dir: str, data_root: str, sigma_s: float, jitter_s: float,
            p_max: int = 10, bootstrap_n: int = 1000, fs: float = 500.0,
            nperseg: int = 512, stride: int = 64, past_len: int = 384,
            log: Optional[Callable] = None, jsonl=None,
            skip_gc: bool = False) -> Dict:
    """skip_gc=True：跳过阶段 1/2（GC 对比阻塞口径，D1），PSD + 空间结构归因照跑（default-off）。"""
    log = log or print
    run_dir = os.path.abspath(run_dir)
    eval_dir = os.path.join(run_dir, "eval")
    run_id = os.path.basename(run_dir.rstrip("/"))
    from common.visualizer import Visualizer  # 延迟导入（图仅此处用）
    viz = Visualizer(os.path.join(run_dir, "figs"))
    _ = jitter_s  # K_s 对比用加 jitter 前原始核（口径登记），jitter_s 仅保签名兼容

    d = np.load(os.path.join(eval_dir, "gc_input.npz"))
    gen, real = d["gen"], d["real"]                    # (N, C, F) f32
    win_subj, win_start = d["win_subj"], d["win_start"]
    N, C, F = real.shape
    subjects = np.unique(win_subj)
    subj_masks = [win_subj == sid for sid in subjects]
    log(f"[gc] 输入 {real.shape} | 受试者 {len(subjects)} 例 | future-only 口径 | "
        f"p_max={p_max}")

    # ---- 阶段 1/2：BIC + 两组条件 GC（skip_gc=True 时整体跳过；GC 对比阻塞口径 D1）----
    gc_stats = "skipped(blocked)"
    gc_fig = None
    if skip_gc:
        log("[gc] skip_gc=True：跳过 BIC/VAR 条件 GC/一致性/bootstrap（genGC vs real GC 对比暂时"
            "阻塞，D1 拍板）；不落任何 GC 产物；PSD 与空间结构归因照跑")
        if jsonl:
            jsonl.write({"stage": "gc", "gc": "skipped(blocked)",
                         "note": "GC 对比阻塞（D1）；PSD/空间结构归因照跑"})
    else:
        t0 = time.time()
        pieces_max_real = [gram_pieces(real[m], p_max, row_from=p_max) for m in subj_masks]
        p_star, bic_tab = bic_from_gram(sum_grams(pieces_max_real), log=log)
        log(f"[gc] BIC 完成 {time.time() - t0:.1f}s | p*={p_star}")

        # ---- 阶段 2：两组条件 GC（受试者级 Gram，精确行集 t∈[p*,F)）----
        pieces_real, pieces_gen = [], []
        for si, m in enumerate(subj_masks):
            pieces_real.append(gram_pieces(real[m], p_star))
            pieces_gen.append(gram_pieces(gen[m], p_star))
            if (si + 1) % 5 == 0 or si == len(subj_masks) - 1:
                el = time.time() - t0
                log(f"[gc-gram] 受试者 {si + 1}/{len(subj_masks)} | {el:.0f}s | "
                    f"ETA {el / (si + 1) * (len(subj_masks) - si - 1):.0f}s")
        del pieces_max_real
        out_r = conditional_gc(sum_grams(pieces_real), p_star)
        out_g = conditional_gc(sum_grams(pieces_gen), p_star)
        np.save(os.path.join(eval_dir, "gc_real.npy"), out_r["gc"])
        np.save(os.path.join(eval_dir, "gc_gen.npy"), out_g["gc"])
        np.savez(os.path.join(eval_dir, "gc_F_p_real.npz"), Fmat=out_r["F"], pmat=out_r["p"])
        np.savez(os.path.join(eval_dir, "gc_F_p_gen.npz"), Fmat=out_g["F"], pmat=out_g["p"])

        xr, xg = offdiag_flat(out_r["gc"]), offdiag_flat(out_g["gc"])
        r, p_r = r_and_p(xr, xg)
        rho, p_rho = r_and_p(rank_transform(xr), rank_transform(xg))
        log(f"[gc] 矩阵一致性：Pearson r={r:.4f} (raw p={p_r:.2e}) | Spearman rho={rho:.4f} "
            f"(raw p={p_rho:.2e}) | n_offdiag={len(xr)}（矩阵元素非独立，p 为朴素口径）")

        qr = bh_fdr(out_r["p"][~np.eye(C, dtype=bool)])
        qg = bh_fdr(out_g["p"][~np.eye(C, dtype=bool)])
        sig_r, sig_g = qr < 0.05, qg < 0.05
        jac = float(np.sum(sig_r & sig_g) / max(np.sum(sig_r | sig_g), 1))
        log(f"[gc] BH-FDR q<0.05 显著边：真实 {int(sig_r.sum())}/{len(qr)} | 生成 {int(sig_g.sum())} | "
            f"Jaccard {jac:.3f}")

        boot = _bootstrap_corr_ci(pieces_real, pieces_gen, p_star, bootstrap_n,
                                  BOOTSTRAP_SEED, log)
        log(f"[gc] bootstrap（受试者级 {bootstrap_n} 次）：r 95% CI {boot['pearson_ci95']} | "
            f"rho 95% CI {boot['spearman_ci95']}")

        gc_stats = {"p_star": p_star, "bic_table": bic_tab, "n_rows": int(out_r["n"]),
                    "pearson_r": round(r, 5), "pearson_raw_p": p_r,
                    "spearman_rho": round(rho, 5), "spearman_raw_p": p_rho,
                    "n_sig_q05": {"real": int(sig_r.sum()), "gen": int(sig_g.sum())},
                    "sig_jaccard": round(jac, 4), "bootstrap": boot,
                    "gc_range": {"real": [float(xr.min()), float(xr.max())],
                                 "gen": [float(xg.min()), float(xg.max())]}}
        if jsonl:
            jsonl.write({"stage": "gc", **gc_stats})

        gc_fig = viz.render("gc_matrix", {
            "gc_real": out_r["gc"], "gc_gen": out_g["gc"],
            "stats": {"pearson_r": r, "spearman_rho": rho, "pearson_ci95": boot["pearson_ci95"]},
        }, f"eval_gc_matrix_{run_id}.png")

    # ---- 阶段 3：PSD（两组完全相同分段与拼接；绝对 512 点网格对齐）----
    t0 = time.time()
    keep = (win_start % (2 * stride)) == 0                # future 段互不重叠且首尾相接
    log(f"[psd] 互不重叠 future 段窗：{int(keep.sum())}/{N}（start ≡ 0 mod 2·stride）| "
        f"nperseg={nperseg} Hann fs={fs}（分辨率 {fs / nperseg:.3f} Hz）")
    acc_r = np.zeros((C, nperseg // 2 + 1))
    acc_g = np.zeros_like(acc_r)
    nseg_total = 0
    freqs = None
    sel_subjects = np.unique(win_subj[keep])
    for si, sid in enumerate(sel_subjects):
        m = keep & (win_subj == sid)
        wi = np.where(m)[0][np.argsort(win_start[m])]     # 按窗序拼接
        drop = (-(past_len + int(win_start[wi[0]]))) % nperseg  # 前导对齐绝对 512 网格
        for acc, arr in ((acc_r, real), (acc_g, gen)):
            series = arr[wi].transpose(1, 0, 2).reshape(C, -1)
            series = series[:, drop:]
            mseg = series.shape[1] // nperseg
            for c in range(C):
                freqs, p_ = welch_psd(series[c], fs, nperseg)
                acc[c] += p_ * mseg
        nseg_total += mseg
        if (si + 1) % 3 == 0 or si == len(sel_subjects) - 1:
            el = time.time() - t0
            log(f"[psd] 受试者 {si + 1}/{len(sel_subjects)} | 段累计 {nseg_total} | {el:.0f}s")
    psd_r, psd_g = acc_r / nseg_total, acc_g / nseg_total
    np.savez(os.path.join(eval_dir, "psd.npz"), freqs=freqs, psd_real=psd_r, psd_gen=psd_g,
             n_segments=nseg_total)
    pos = freqs > 0
    # 汇总口径：全通道平均谱的逐频 |Δlog|（逐通道谱在对数轴的深谷会放大噪声伪差）
    mr_spec, mg_spec = psd_r[:, pos].mean(0), psd_g[:, pos].mean(0)
    logdiff_mean = np.abs(np.log(np.maximum(mr_spec, 1e-30)) - np.log(np.maximum(mg_spec, 1e-30)))
    psd_stats = {"n_segments": int(nseg_total), "resolution_hz": round(float(fs / nperseg), 4),
                 "mean_abs_logdiff": round(float(logdiff_mean.mean()), 5),
                 "peak_hz": {"real": float(freqs[pos][psd_r[:, pos].mean(0).argmax()]),
                             "gen": float(freqs[pos][psd_g[:, pos].mean(0).argmax()])}}
    log(f"[psd] 全通道平均 log-PSD |Δ| 均值 = {psd_stats['mean_abs_logdiff']:.4f} | "
        f"峰频率 real {psd_stats['peak_hz']['real']:.2f} Hz vs gen {psd_stats['peak_hz']['gen']:.2f} Hz")
    if jsonl:
        jsonl.write({"stage": "psd", **psd_stats})
    psd_fig = viz.render("psd", {"freqs": freqs, "psd_real": psd_r, "psd_gen": psd_g},
                         f"eval_psd_{run_id}.png")

    # ---- 阶段 4：零滞后空间结构归因（指标 4）----
    t0 = time.time()
    R_hat = zero_lag_corr(gen)
    R_real = zero_lag_corr(real)
    coords = np.load(os.path.join(data_root, "chan_coords_62.npy"))
    if C != coords.shape[0]:
        log(f"[spatial] 通道数 {C}≠{coords.shape[0]}——按前 {C} 通道坐标重算 K_s（非 62 通道冒烟口径）")
        coords = coords[:C]
    K_s = ricker3d_kernel(coords, sigma_s)          # 加 jitter 前的原始核（对比口径）
    sp = three_way(R_hat, K_s, R_real)
    np.savez(os.path.join(eval_dir, "spatial_corr.npz"), R_hat=R_hat, K_s=K_s, R_real=R_real)
    log(f"[spatial] Frobenius：R̂−K_s {sp['fro_Rhat_Ks']:.3f} | R̂−R {sp['fro_Rhat_R']:.3f} | "
        f"K_s−R {sp['fro_Ks_R']:.3f}；上三角 r：R̂~K_s {sp['tri_r_Rhat_Ks']:.3f} | "
        f"R̂~R {sp['tri_r_Rhat_R']:.3f} | K_s~R {sp['tri_r_Ks_R']:.3f}（{time.time() - t0:.1f}s）")
    if jsonl:
        jsonl.write({"stage": "spatial", **sp})
    sp_fig = viz.render("spatial_corr", {"R_hat": R_hat, "K_s": K_s, "R_real": R_real,
                                         "stats": sp}, f"eval_spatial_corr_{run_id}.png")

    if skip_gc:
        res = {"gc": gc_stats, "psd": psd_stats, "spatial": sp,
               "figures": [psd_fig, sp_fig],
               "arrays": {"psd": "eval/psd.npz", "spatial": "eval/spatial_corr.npz"}}
    else:
        res = {"gc": gc_stats, "psd": psd_stats, "spatial": sp,
               "figures": [gc_fig, psd_fig, sp_fig],
               "arrays": {"gc_real": "eval/gc_real.npy", "gc_gen": "eval/gc_gen.npy",
                          "psd": "eval/psd.npz", "spatial": "eval/spatial_corr.npz"}}
    with open(os.path.join(eval_dir, "gc_psd_results.json"), "w") as fp:
        json.dump(res, fp, indent=2, ensure_ascii=False, default=float)
    log(f"[gc_psd] 结果落 {eval_dir}/gc_psd_results.json + npy | 图 {res['figures']}"
        f"{'（GC 对比阻塞：无 gc 产物/图）' if skip_gc else ''}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--p-max", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--nperseg", type=int, default=512)
    args = ap.parse_args()
    run_dir = os.path.join("/root/npi/outputs/gpfm-mamba-f0", args.run_id)
    run_all(run_dir=run_dir, data_root="/root/autodl-tmp/edra_process_v4",
            sigma_s=0.080, jitter_s=1e-3, p_max=args.p_max, bootstrap_n=args.bootstrap,
            fs=args.fs, nperseg=args.nperseg)


if __name__ == "__main__":
    main()
