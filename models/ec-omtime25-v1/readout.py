"""readout.py — slope/AUC/r* 三 readout + D̃ 曲线 + 有效路径 + 契约 npz
（10-18-ec-omtime25-f0；沿 perturb-ec-v1/readout.py 结构自实现，读自身 config）。

输入 dbar 曲线一律与 R_GRID（9 点含 0）首尾对齐：**索引 0 ≡ r=0 ≡ D̄=0**（CRN 恒等
按构造，不采样）。主 readout = 低剂量斜率（天然大=强免翻转；evaluation-calibers）。
"""

from __future__ import annotations

import numpy as np

import candidates as CD

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def slope_readout(dbar, seg=None):
    """主 readout：SLOPE_SEG（5 点含原点）OLS 斜率。dbar (..., 9) → (...)。"""
    SC = CD.load_config()
    x = np.asarray(SC.SLOPE_SEG if seg is None else seg, dtype=np.float64)
    y = np.asarray(dbar)[..., : len(x)]
    xc = x - x.mean()
    yc = y - y.mean(axis=-1, keepdims=True)
    return (xc * yc).sum(axis=-1) / (xc * xc).sum()


def auc_readout(dbar):
    """次 readout：全网格梯形面积 / R_MAX。dbar (..., 9) → (...)。"""
    SC = CD.load_config()
    r = np.asarray(SC.R_GRID, dtype=np.float64)
    return _trapz(np.asarray(dbar), r, axis=-1) / SC.R_MAX


def dtil_curves(dbar9):
    """τ v2 相对自响应曲线：D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r)（分母 = 靶自响应）。

    输入 dbar9 (62, 9, 62) 索引 [A, r, B]；输出 (62, 62, 9) 索引 [A, B, r]。
    r=0 槽置 0（D̄(0)=0 按构造）；A=B 处 D̃≡1（r>0 槽，仅约定不进判定）。
    判读纪律：分母靶自响应偏小会放大比值，D̃ 数值不作跨靶比较。"""
    SC = CD.load_config()
    dbar9 = np.asarray(dbar9, dtype=np.float64)
    g = len(SC.R_GRID)
    n = dbar9.shape[0]
    assert dbar9.ndim == 3 and dbar9.shape == (n, g, n), \
        f"dbar9 形状 {dbar9.shape} 须为 (n, {g}, n)（[A, r, B]）"
    dtil = np.zeros((n, n, g), dtype=np.float64)
    idx = np.arange(n)
    for j in range(1, g):
        Mj = dbar9[:, j, :]
        selfj = Mj[idx, idx]
        assert (selfj > 0).all(), f"r 档 {j} 存在零自响应（D̃ 除零）"
        dtil[:, :, j] = Mj / selfj[None, :]
    dtil[idx, idx, 1:] = 1.0
    return dtil


def rstar_readout(dtil, tau):
    """r* = min{r ∈ R_GRID : D̃(r) ≥ τ}；网格内未达 → 右删失编码 R_NEXT。dtil (..., 9)。"""
    SC = CD.load_config()
    hit = np.asarray(dtil) >= tau
    first = np.argmax(hit, axis=-1)
    reached = hit.any(axis=-1)
    return np.where(reached, np.asarray(SC.R_GRID)[first], SC.R_NEXT)


def flip_rstar(rstar):
    """rstar → strength（契约大=强）：R_NEXT − r*；删失 → 0 最弱。"""
    return CD.load_config().R_NEXT - np.asarray(rstar)


def effective_paths(rstar):
    """有效路径集：{(A, B) : A ≠ B, r* ≤ R_MAX}。返回有序 [(a, b), ...]。"""
    SC = CD.load_config()
    rs = np.asarray(rstar)
    m = (rs <= SC.R_MAX) & ~np.eye(rs.shape[0], dtype=bool)
    return [(int(a), int(b)) for a, b in zip(*np.nonzero(m))]


def save_ec_pert_npz(path, ec, readout: str, subject_ids):
    """契约 npz 落盘（比较器接口预留）：ec (62,62) / readout∈{slope,auc,rstar} /
    subject_ids（id 数组或 'global'）/ orientation='larger_is_stronger'（rstar 已 flip）。"""
    SC = CD.load_config()
    assert readout in ("slope", "auc", "rstar"), f"未知 readout {readout!r}"
    ec = np.asarray(ec)
    assert ec.shape == (SC.N_SRC, SC.N_SRC), f"ec 形状 {ec.shape} != (62,62)"
    np.savez(path, ec=ec, readout=readout, subject_ids=subject_ids,
             orientation=SC.ORIENTATION)
