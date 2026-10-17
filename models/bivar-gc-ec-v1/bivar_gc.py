"""bivar_gc —— 成对双变量 GC 核心（numpy-only，全部 float64；design §2.2）。

口径（用户三轮拍板落定，design §2.2 为技术规格权威）：
- **行集对齐惯例**：BIC 全候选阶共用行集 row_from=P_MAX（嵌套行集，与 baseline-ec-v1
  「对齐行集 row_from=p_max」同惯例，消除各阶样本数差异）；
- **滞后设计列序** = [截距, ch0 lag1..P_MAX, ch1 lag1..P_MAX, ...]（通道主序；
  p'<p 的模型为列子集 → BIC 嵌套比较合法）；
- **逐无序对 {A,B}**：从全通道滞后 Gram slice 出 [截距, A lag1..p, B lag1..p]
  （2p+1 维；p=P_MAX=24 时 49 维，即 design §2.2「≤49 维小系统」）的二通道 VAR(p)
  OLS（两方程一次解）→ SSR_u 二次型 + restricted（同阶单通道 AR，[截距, 自身 lag1..p]
  (p+1) 维）→ BIC 选阶 p*({A,B}) → p* 下同一次拟合双向读出；
- **BIC(p) = ln|Σ̂(p)| + (K²·p)·ln(n)/n**，K=2（Σ̂=2×2 残差协差 (1/n)·SSR 对称化；
  截距跨阶同数不进 argmin 比较）；
- **GC(A→B) = ln(SSR_r^B / SSR_u^B)**（restricted = B 仅回归自身过去；unrestricted =
  B 回归 A、B 共同过去）；矩阵行=源/列=靶/对角按构造=0；
- **样本内必然非负**（OLS 残差 SSR 单调性）：offdiag GC≥0 作为实现正确性断言，
  负值=实现错误即刻失败（与 ridge 条件 GC 允许微负不同，产物 caliber 须注明区分）；
- **SSR 二次型** = YtY − 2BᵀZy + BᵀGB（与 Gram slice 一致，两路径同式）。

naive 参考实现（naive_pair_fit / naive_bivar_gc_matrix）：逐对逐阶直接
np.linalg.lstsq 设计矩阵拟合（SVD 路径、残差直算 SSR），公式与快路径逐式同构，
供等价性单测（rtol=1e-10）与人工对照；主路径一律走 Gram-slice。

分段输入：X 支持 (C,T) 连续记录与 (N,C,F) 分段（split-half 段式——段内设计不跨段，
段间行堆叠），build_gram 统一处理。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md 惯例：必须在 numpy 之前；setdefault 可外部覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import math               # noqa: E402
from typing import Dict, Optional, Sequence, Tuple  # noqa: E402

import numpy as np        # noqa: E402

K_PAIR = 2                # 二通道 VAR 的通道数（BIC 惩罚项 K²·p 的 K）


# ================= 设计矩阵与 Gram =================

def _lag_cols(c: int, p: int, p_max: int) -> np.ndarray:
    """全宽设计中通道 c 前 p 个滞后列的 0 基列号（通道主序：1 + c*p_max + (l-1)）。"""
    assert 1 <= p <= p_max and 0 <= c
    return 1 + c * p_max + np.arange(p)


def _as_segments(X: np.ndarray) -> np.ndarray:
    """(C,T) 连续 → (1,C,T)；(N,C,F) 分段原样。float64 化。"""
    Xp = np.asarray(X, dtype=np.float64)
    return Xp[None] if Xp.ndim == 2 else Xp


def var_design(X: np.ndarray, p_max: int, row_from: Optional[int] = None,
               t0: Optional[int] = None, t1: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """全通道滞后设计（快路径素材 + 诊断用；naive 路径不经此函数）。

    X: (C,T) 连续或 (N,C,F) 分段（段内设计不跨段）；行集 t∈[max(row_from,p_max), F)
    内的窗 [t0,t1)（缺省全行集；build_gram 分块用）。
    Z 列序 = [截距, ch0 lag1..p_max, ch1 lag1..p_max, ...]（通道主序；float64）。
    返回 Z (n, 1+C·p_max), Y (n, C)；行序 = 段主序 × 段内 t 升序。
    """
    Xp = _as_segments(X)
    N, C, F = Xp.shape
    r0 = p_max if row_from is None else row_from
    assert r0 >= p_max, f"row_from={r0} < p_max={p_max}（滞后越界）"
    t_lo = r0 if t0 is None else max(t0, r0)
    t_hi = F if t1 is None else min(t1, F)
    assert t_hi > t_lo, f"空行集：t∈[{t_lo},{t_hi})"
    n = N * (t_hi - t_lo)
    Z = np.ones((n, 1 + C * p_max), dtype=np.float64)
    for c in range(C):
        for l in range(1, p_max + 1):
            Z[:, 1 + c * p_max + (l - 1)] = Xp[:, c, t_lo - l: t_hi - l].reshape(n)
    Y = Xp[:, :, t_lo: t_hi].transpose(0, 2, 1).reshape(n, C)
    return Z, Y


def build_gram(X: np.ndarray, p_max: int, row_from: Optional[int] = None,
               row_chunk: int = 8192) -> Dict:
    """全通道滞后 Gram 三件套（fast 路径核心：每例一次构建，逐对逐阶 slice 复用）。

    G = ZᵀZ (1+C·p_max)²、Zy = ZᵀY、YtY = YᵀY、n = 行数；float64，按行块累积控内存
    （连续长段按绝对 t 窗分块、短段按段组累积——块内 Z 瞬时 ≤ row_chunk 行）。
    row_from 缺省 p_max（BIC 对齐行集惯例；design §2.2）。
    """
    Xp = _as_segments(X)
    N, C, F = Xp.shape
    r0 = p_max if row_from is None else row_from
    assert r0 >= p_max
    R = F - r0
    assert R > 0, f"段长 {F} 不足（row_from={r0}）"
    if R > row_chunk:      # 长段（连续记录）：绝对 t 窗分块
        items = [(slice(0, N), tlo, min(tlo + row_chunk, F))
                 for tlo in range(r0, F, row_chunk)]
    else:                  # 短段（split-half 分段）：按段组累积
        segs = max(1, row_chunk // R)
        items = [(slice(s0, min(s0 + segs, N)), r0, F)
                 for s0 in range(0, N, segs)]
    G = np.zeros((1 + C * p_max, 1 + C * p_max))
    Zy = np.zeros((1 + C * p_max, C))
    YtY = np.zeros((C, C))
    n_rows = 0
    for seg_sl, tlo, thi in items:
        Z, Y = var_design(Xp[seg_sl], p_max, row_from=r0, t0=tlo, t1=thi)
        G += Z.T @ Z
        Zy += Z.T @ Y
        YtY += Y.T @ Y
        n_rows += Z.shape[0]
    return {"G": G, "Zy": Zy, "YtY": YtY, "n": float(n_rows),
            "p_max": p_max, "C": C}


# ================= Gram-slice 快路径（主路径） =================

def _unrestricted_slice(gram: Dict, a: int, b: int, p: int) -> np.ndarray:
    """二通道 VAR(p) unrestricted 子问题（Gram-slice → OLS 一次解两方程）。

    列 = [截距, A lag1..p, B lag1..p]（2p+1 维）；返回 SSR_u (2,2)
    （二次型 YtY−2BᵀZy+BᵀGB，含残差交叉项；索引 0=通道 a、1=通道 b）。
    """
    p_max = gram["p_max"]
    cols = np.concatenate(([0], _lag_cols(a, p, p_max), _lag_cols(b, p, p_max)))
    G = gram["G"][np.ix_(cols, cols)]
    Zy = gram["Zy"][cols][:, [a, b]]
    YtY = gram["YtY"][np.ix_([a, b], [a, b])]
    B = np.linalg.solve(G, Zy)                       # (2p+1, 2) 两方程一次解
    return YtY - 2.0 * (B.T @ Zy) + B.T @ G @ B


def _restricted_slice(gram: Dict, c: int, p: int) -> float:
    """restricted 子问题：通道 c 仅回归自身过去（[截距, 自身 lag1..p]，p+1 维）。"""
    p_max = gram["p_max"]
    cols = np.concatenate(([0], _lag_cols(c, p, p_max)))
    G = gram["G"][np.ix_(cols, cols)]
    zy = gram["Zy"][cols, c]
    yty = gram["YtY"][c, c]
    B = np.linalg.solve(G, zy)
    return float(yty - 2.0 * (B @ zy) + B @ G @ B)


def pair_fit(gram: Dict, a: int, b: int, p_grid: Sequence[int],
             p_max: Optional[int] = None) -> Dict:
    """逐无序对 {A,B} 逐阶拟合：BIC 曲线 → p* → 同阶双向 ln(SSR) 读出（主路径）。

    返回 {"p_grid", "bic" (P,), "logdet" (P,), "ssr_u" (P,2,2), "ssr_r" (P,2),
    "p_star", "gc_ab", "gc_ba"}；gc_ab = GC(A→B) = ln(SSR_r^B/SSR_u^B)。
    p* 取 BIC argmin（并列取最小 p；全 inf 则断言失败）。
    """
    p_max = gram["p_max"] if p_max is None else p_max
    n = gram["n"]
    Ps = list(p_grid)
    Pn = len(Ps)
    bic = np.full(Pn, np.inf)
    logdet = np.full(Pn, np.nan)
    ssr_u = np.zeros((Pn, 2, 2))
    ssr_r = np.zeros((Pn, 2))
    for i, p in enumerate(Ps):
        Su = _unrestricted_slice(gram, a, b, p)
        ssr_u[i] = Su
        ssr_r[i, 0] = _restricted_slice(gram, a, p)
        ssr_r[i, 1] = _restricted_slice(gram, b, p)
        Sig = (Su + Su.T) / 2.0 / n                   # Σ̂=(1/n)·SSR 对称化（2×2）
        sign, ld = np.linalg.slogdet(Sig)
        logdet[i] = ld
        if sign > 0:
            bic[i] = ld + (K_PAIR ** 2) * p * math.log(n) / n   # 截距不进惩罚
    i_star = int(np.argmin(bic))
    assert np.isfinite(bic[i_star]), \
        f"pair ({a},{b}): BIC 全候选阶非有限（Σ̂ 非 PD？）"
    p_star = int(Ps[i_star])
    sru, srr = ssr_u[i_star], ssr_r[i_star]
    assert sru[0, 0] > 0 and sru[1, 1] > 0, f"pair ({a},{b}): SSR_u ≤ 0"
    assert srr[0] >= sru[0, 0] and srr[1] >= sru[1, 1], \
        f"pair ({a},{b}): SSR_r < SSR_u（OLS 单调性违约=实现错误）"
    return {"p_grid": tuple(Ps), "bic": bic, "logdet": logdet,
            "ssr_u": ssr_u, "ssr_r": ssr_r, "p_star": p_star,
            "gc_ab": float(np.log(srr[1] / sru[1, 1])),   # A→B：B 方程加 A 滞后的改进
            "gc_ba": float(np.log(srr[0] / sru[0, 0]))}


def bivar_gc_matrix(X: np.ndarray, p_max: int = 24,
                    p_grid: Optional[Sequence[int]] = None,
                    row_from: Optional[int] = None) -> Dict:
    """主入口：全通道 Gram 一次构建 → 逐无序对 {A,B} BIC 选阶 + 双向 ln(SSR) 读出。

    X: (C,T) 连续或 (N,C,F) 分段（段内设计不跨段）。row_from 缺省 p_max（对齐行集）。
    返回 {"ec" (C,C) 行=源/列=靶/对角=0（float64）, "orders" (C,C) int64 对称登记
    （{A,B} 同阶）, "n" 行数, "p_grid"}。
    断言（实现正确性契约）：全有限 / 对角=0 / offdiag ≥ 0（负值=实现错误即刻失败）。
    """
    if p_grid is None:
        p_grid = tuple(range(1, p_max + 1))
    gram = build_gram(X, p_max, row_from=row_from)
    C = gram["C"]
    ec = np.zeros((C, C))
    orders = np.zeros((C, C), dtype=np.int64)
    for a in range(C - 1):
        for b in range(a + 1, C):
            r = pair_fit(gram, a, b, p_grid, p_max)
            ec[a, b] = r["gc_ab"]        # 行=源 a，列=靶 b
            ec[b, a] = r["gc_ba"]
            orders[a, b] = orders[b, a] = r["p_star"]
    assert np.isfinite(ec).all(), "EC 矩阵含非有限值"
    assert float(np.abs(np.diag(ec)).max()) == 0.0, "EC 对角非零（口径违约）"
    od = ec[~np.eye(C, dtype=bool)]
    assert (od >= 0).all(), \
        f"offdiag GC < 0（min={od.min():.3e}）：样本内 OLS SSR 单调性必然非负，负值=实现错误"
    return {"ec": ec, "orders": orders, "n": gram["n"], "p_grid": tuple(p_grid)}


# ================= naive 参考实现（等价性单测与对照；不进主路径） =================

def _naive_design(Xp: np.ndarray, chans: Sequence[int], p: int, r0: int) -> np.ndarray:
    """直接从原始数据组装 [截距, (各通道) lag1..p] 设计（列序随 chans；段内不跨段）。"""
    N, _, F = Xp.shape
    n = N * (F - r0)
    Z = np.ones((n, 1 + len(chans) * p), dtype=np.float64)
    for j, c in enumerate(chans):
        for l in range(1, p + 1):
            Z[:, 1 + j * p + (l - 1)] = Xp[:, c, r0 - l: F - l].reshape(n)
    return Z


def naive_pair_fit(X: np.ndarray, a: int, b: int, p_grid: Sequence[int],
                   row_from: Optional[int] = None) -> Dict:
    """naive 参考：逐对逐阶直接 np.linalg.lstsq 设计矩阵拟合，残差直算 SSR。

    公式与 pair_fit 逐式同构（Σ̂=(1/n)SSR 对称化 / BIC=ln|Σ̂|+K²p·ln n/n /
    p* 同阶 restricted 读出），仅求解路径不同（SVD lstsq + 残差直算 vs
    Gram-slice 正规方程 + 二次型）——等价性单测 T3 的对照基准。
    行集同 row_from（缺省 max(p_grid)，与主路径对齐行集一致）。
    """
    Xp = _as_segments(X)
    r0 = max(p_grid) if row_from is None else row_from
    N, C, F = Xp.shape
    Y = Xp[:, :, r0:].transpose(0, 2, 1).reshape(-1, C)
    n = float(Y.shape[0])
    Ps = list(p_grid)
    Pn = len(Ps)
    bic = np.full(Pn, np.inf)
    logdet = np.full(Pn, np.nan)
    ssr_u = np.zeros((Pn, 2, 2))
    ssr_r = np.zeros((Pn, 2))
    for i, p in enumerate(Ps):
        Zu = _naive_design(Xp, (a, b), p, r0)
        Bu, *_ = np.linalg.lstsq(Zu, Y[:, [a, b]], rcond=None)
        eu = Y[:, [a, b]] - Zu @ Bu
        Su = eu.T @ eu
        ssr_u[i] = Su
        for k, c in enumerate((a, b)):
            Zr = _naive_design(Xp, (c,), p, r0)
            Br, *_ = np.linalg.lstsq(Zr, Y[:, c], rcond=None)
            er = Y[:, c] - Zr @ Br
            ssr_r[i, k] = er @ er
        Sig = (Su + Su.T) / 2.0 / n
        sign, ld = np.linalg.slogdet(Sig)
        logdet[i] = ld
        if sign > 0:
            bic[i] = ld + (K_PAIR ** 2) * p * math.log(n) / n
    i_star = int(np.argmin(bic))
    assert np.isfinite(bic[i_star]), f"naive pair ({a},{b}): BIC 全候选阶非有限"
    sru, srr = ssr_u[i_star], ssr_r[i_star]
    return {"p_grid": tuple(Ps), "bic": bic, "logdet": logdet,
            "ssr_u": ssr_u, "ssr_r": ssr_r, "p_star": int(Ps[i_star]),
            "gc_ab": float(np.log(srr[1] / sru[1, 1])),
            "gc_ba": float(np.log(srr[0] / sru[0, 0]))}


def naive_bivar_gc_matrix(X: np.ndarray, p_max: int = 24,
                          p_grid: Optional[Sequence[int]] = None,
                          row_from: Optional[int] = None) -> Dict:
    """naive 全矩阵（逐对调 naive_pair_fit；慢路径，仅单测与对照用）。"""
    if p_grid is None:
        p_grid = tuple(range(1, p_max + 1))
    Xp = _as_segments(X)
    C = Xp.shape[1]
    ec = np.zeros((C, C))
    orders = np.zeros((C, C), dtype=np.int64)
    for a in range(C - 1):
        for b in range(a + 1, C):
            r = naive_pair_fit(Xp, a, b, p_grid, row_from=row_from)
            ec[a, b] = r["gc_ab"]
            ec[b, a] = r["gc_ba"]
            orders[a, b] = orders[b, a] = r["p_star"]
    return {"ec": ec, "orders": orders, "n": float(Xp.shape[2] - (row_from or p_max))
            * Xp.shape[0], "p_grid": tuple(p_grid)}


# ================= 数值监控（smoke 条件数登记；design §2.2/§5-3） =================

def gram_pair_cond(gram: Dict, a: int, b: int, p: int) -> float:
    """对子问题 Gram slice（[截距, A lag1..p, B lag1..p]）的 2-范数条件数。

    OLS 数值条件监控（预注册备选 λ_rel=1e-4 对角加载的依据登记；二通道 2p+1≤49
    回归元 + n≈3 万行，与全模型 62ch ~1e18 条件数情形不同构——design §2.2）。
    """
    p_max = gram["p_max"]
    cols = np.concatenate(([0], _lag_cols(a, p, p_max), _lag_cols(b, p, p_max)))
    return float(np.linalg.cond(gram["G"][np.ix_(cols, cols)]))


def subset_cond_stats(gram: Dict, chan_idx: Sequence[int], p: int) -> Dict:
    """通道子集内全部无序对的 pair-Gram 条件数统计（smoke 阶段 8ch 子集预演用）。"""
    vals = [gram_pair_cond(gram, a, b, p)
            for i, a in enumerate(chan_idx) for b in list(chan_idx)[i + 1:]]
    v = np.asarray(vals)
    return {"min": float(v.min()), "median": float(np.median(v)),
            "max": float(v.max()), "n_pairs": int(v.size)}
