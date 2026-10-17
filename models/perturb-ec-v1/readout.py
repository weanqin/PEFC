"""readout.py — slope/AUC/r* 三 readout + r* 翻转 + 有效路径集 + load_ec_pert 契约 npz
（design §2.4；evaluation-calibers.md Study II 节：主 = 低剂量斜率，天然大=强免翻转）。

输入 D̄ 曲线一律与 R_GRID（9 点含 0）首尾对齐：**索引 0 ≡ r=0 ≡ D̄=0**（CRN 恒等按构造，
不采样）。R_NEXT / 翻转偏移 / 斜率段一律由 R_GRID 推导（禁字面量）——档位调整权
（evaluation-calibers 变更纪律）启用时自动一致（design §7-6）。
"""

from __future__ import annotations

import numpy as np

import perturb as PB

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.x 改名兼容


def slope_readout(dbar, seg=None):
    """主 readout：低剂量段 OLS 斜率（普通最小二乘含截距；design §2.4）。

    段默认 R_GRID[:5] = {0, 0.01, 0.02, 0.04, 0.08}（5 点含原点）；敏感性段
    R_GRID[:4] 经 seg=PC.SLOPE_SEG_SENS 传入。dbar (..., 9) → (...)；天然大=强。
    """
    PC = PB.load_config()
    x = np.asarray(PC.SLOPE_SEG if seg is None else seg, dtype=np.float64)
    y = np.asarray(dbar)[..., : len(x)]
    xc = x - x.mean()
    yc = y - y.mean(axis=-1, keepdims=True)
    return (xc * yc).sum(axis=-1) / (xc * xc).sum()


def auc_readout(dbar):
    """次 readout：全网格梯形面积 / R_MAX 归一（design §2.4；大=强，判读随附 §2.1
    高剂量外推 caveat）。dbar (..., 9) → (...)。"""
    PC = PB.load_config()
    r = np.asarray(PC.R_GRID, dtype=np.float64)
    return _trapz(np.asarray(dbar), r, axis=-1) / PC.R_MAX


def rstar_readout(dbar, tau):
    """r* = min{r ∈ R_GRID : D̄(A→B)(r) ≥ τ}（design §2.4）。

    网格内未达 → 右删失，编码 r* = R_NEXT（下一档，由 R_GRID 推导禁字面量）；
    敏感性口径随 q90/q99 τ 重算（B1+ 登记）。dbar (..., 9) → (...)。
    """
    PC = PB.load_config()
    hit = np.asarray(dbar) >= tau
    first = np.argmax(hit, axis=-1)
    reached = hit.any(axis=-1)
    return np.where(reached, np.asarray(PC.R_GRID)[first], PC.R_NEXT)


def flip_rstar(rstar):
    """rstar → strength 翻转（进比较器前；契约大=强）：strength = R_NEXT − r*。

    右删失（r* = R_NEXT）→ 0 最弱；偏移 = R_NEXT 由 R_GRID 推导，禁字面量（design §2.4）。
    """
    return PB.load_config().R_NEXT - np.asarray(rstar)


def effective_paths(rstar):
    """有效路径集（design §2.4）：{(A, B) : A ≠ B, r*_{A→B} ≤ R_MAX}（τ 主口径判定；
    删失边编码 R_NEXT > R_MAX 自动排除）。返回有序 [(a, b), ...]（行=源 a、列=靶 b）。"""
    PC = PB.load_config()
    m = (np.asarray(rstar) <= PC.R_MAX) & ~np.eye(PC.N_SRC, dtype=bool)
    return [(int(a), int(b)) for a, b in zip(*np.nonzero(m))]


def dtil_curves(dbar9):
    """τ v2 相对自响应曲线（design §2.3 v2，Run 004）：D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r)。

    输入 dbar9 (62, 9, 62) 索引 [A, r, B]（主 Run dbar_median_s*.npy 原生轴序）；输出
    (62, 62, 9) 索引 [A, B, r]（readout 契约 r 末维）。分母 = 靶通道自响应 D̄_{B→B}(r)
    （dbar9[B, j, B]，同 r 列对角）；r=0 槽置 0（D̄(0)=0 按构造，与 Run 002 预览实现
    一致）；A=B 处 D̃≡1（r>0 槽，仅约定不进路径判定）。判读纪律（design §2.3）：
    分母靶自响应偏小会放大比值，不将 D̃ 数值作跨靶比较。
    """
    dbar9 = np.asarray(dbar9, dtype=np.float64)
    PC = PB.load_config()
    n, g = PC.N_SRC, len(PC.R_GRID)
    assert dbar9.shape == (n, g, n), f"dbar9 形状 {dbar9.shape} != ({n},{g},{n})"
    dtil = np.zeros((n, n, g), dtype=np.float64)
    idx = np.arange(n)
    for j in range(1, g):                                  # r=0 槽保持 0
        Mj = dbar9[:, j, :]                                # (A, B)
        selfj = Mj[idx, idx]                               # D̄_{B→B}(r_j)
        assert (selfj > 0).all(), f"r 档 {j} 存在零自响应（D̃ 除零）"
        dtil[:, :, j] = Mj / selfj[None, :]
    return dtil


def save_ec_pert_npz(path, ec, readout: str, subject_ids):
    """load_ec_pert 契约 npz 落盘（比较器 = baseline-ec-v1 compare.py，**子进程/文件面
    交换**，进程内不 import；字段按 design §5 抄写）：

    - `ec` (62, 62)，行=源、列=靶；
    - `readout` ∈ {slope, auc, rstar}；
    - `subject_ids`（受试者 id 数组或 'global'）；
    - `orientation` = 'larger_is_stronger'（rstar 须调用方先 flip_rstar 再存）。
    """
    PC = PB.load_config()
    assert readout in ("slope", "auc", "rstar"), f"未知 readout {readout!r}"
    ec = np.asarray(ec)
    assert ec.shape == (PC.N_SRC, PC.N_SRC), f"ec 形状 {ec.shape} != (62,62)"
    np.savez(path, ec=ec, readout=readout, subject_ids=subject_ids,
             orientation=PC.ORIENTATION)
