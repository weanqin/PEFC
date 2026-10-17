"""split_half —— 双变量 GC 参考自身的 split-half 噪声天花板（核心函数；design §2.3）。

段式复刻 baseline-ec-v1 split_half：段长 250 样本（2 s @125 Hz）；偶段→半 A、
奇段→半 B、尾段丢弃；**段内设计不跨段**（每半以 (n_seg, C, seg) 多段窗喂
bivar_gc.build_gram——段内滞后行 t≥P_MAX..seg-1，段间行堆叠）。
每半**独立同口径**重跑双变量 GC（含逐对 BIC）→ 两半 offdiag 3782 有向边
Spearman ρ（逐例）→ 天花板 = mean±sd（ddof=1，48 例；B3 聚合在 run.py stage_split）。

定位：参考自身的噪声天花板登记，后续任何比较可直接取用作 ρ̄/ceiling 分母。
本阶段（B1）只实现核心函数，不跑数据。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md 惯例：必须在 numpy 之前；setdefault 可外部覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util    # noqa: E402
import sys               # noqa: E402
from pathlib import Path # noqa: E402
from typing import Dict, Optional  # noqa: E402

import numpy as np               # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_HERE = Path(__file__).resolve().parent

# ---- 本变体 config 唯一名装载（design §1 四行配方；先注册后 exec）----
if "bgec_config" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location("bgec_config", str(_HERE / "config.py"))
    _C = importlib.util.module_from_spec(_SPEC)
    sys.modules["bgec_config"] = _C          # 先注册后 exec
    _SPEC.loader.exec_module(_C)
import bgec_config as C                      # noqa: E402

import bivar_gc  # noqa: E402


def build_halves(x: np.ndarray, seg: Optional[int] = None) -> Dict:
    """(C,T) → {'A': (nA,C,seg), 'B': (nB,C,seg), 'n_seg'}（偶段A/奇段B/尾巴丢弃；复刻 baseline 段式）。"""
    seg = C.SEG_SAMPLES if seg is None else seg
    Cn, T = x.shape
    nseg = T // seg
    segs = x[:, : nseg * seg].reshape(Cn, nseg, seg).transpose(1, 0, 2)   # (nseg,C,seg)
    return {"A": segs[0::2].copy(), "B": segs[1::2].copy(), "n_seg": int(nseg)}


def offdiag_flat(m: np.ndarray) -> np.ndarray:
    """有向边展平（行=源，行优先序；非三角）。62×62 → 3782（baseline compare 同式）。"""
    m = np.asarray(m)
    return np.ravel(m[~np.eye(m.shape[0], dtype=bool)])


def half_ec(segs: np.ndarray, p_max: Optional[int] = None) -> Dict:
    """半 → 同口径双变量 GC（分段输入：段内设计不跨段；含逐对 BIC）。"""
    p_max = C.P_MAX if p_max is None else p_max
    return bivar_gc.bivar_gc_matrix(segs, p_max=p_max)


def ceiling_for_subject(x: np.ndarray, p_max: Optional[int] = None) -> Dict:
    """逐例 split-half 天花板：两半同口径 EC 的 offdiag Spearman ρ。

    返回 {"sh_rho", "n_seg", "ec_a", "ec_b", "orders_a", "orders_b", "n_a", "n_b"}。
    """
    halves = build_halves(x)
    ra = half_ec(halves["A"], p_max)
    rb = half_ec(halves["B"], p_max)
    rho = float(spearmanr(offdiag_flat(ra["ec"]), offdiag_flat(rb["ec"])).statistic)
    return {"sh_rho": rho, "n_seg": halves["n_seg"],
            "ec_a": ra["ec"], "ec_b": rb["ec"],
            "orders_a": ra["orders"], "orders_b": rb["orders"],
            "n_a": ra["n"], "n_b": rb["n"]}
