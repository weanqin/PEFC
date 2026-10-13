"""split_half —— 噪声天花板：奇偶交错 2 s 段两半，同口径 EC 的一致性。

口径（design §5）：段长 250 样本（2 s @125 Hz）；偶段→半 A、奇段→半 B；
各半以 (n_seg, C, seg_len) 多段窗喂 Gram（row_from=p，段内设计不跨段=无拼接伪迹）；
与主口径共用 p_shared。天花板^{(m)} = mean_s Spearman(offdiag(A), offdiag(B)) ± sd。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr

import config as C
from ec_refs import conditional_gc_matrix, gpdc_squared, ridge_fit


def build_halves(x: np.ndarray, seg: int = C.SPLIT_SEG_SAMPLES) -> Dict:
    """(C,T) → {'A': (nA,C,seg), 'B': (nB,C,seg)}（偶段 A / 奇段 B，尾巴余数丢弃）。"""
    Cc, T = x.shape
    nseg = T // seg
    segs = x[:, :nseg * seg].reshape(Cc, nseg, seg).transpose(1, 0, 2)   # (nseg,C,seg)
    return {"A": segs[0::2].copy(), "B": segs[1::2].copy()}


def _half_ec(segs: np.ndarray, p: int) -> Dict:
    from gc_psd import gram_pieces  # 已由 ec_refs 引入 v1 路径
    gram = gram_pieces(segs.astype(np.float64), p, row_from=p)
    gc = conditional_gc_matrix(gram)
    co = ridge_fit(gram)
    gp = gpdc_squared(co["B"], co["Sigma"])
    return {"gc": gc, "gpdc": gp["broadband"]}


def ceiling_for_subject(x: np.ndarray, p: int) -> Dict:
    halves = build_halves(x)
    ea, eb = _half_ec(halves["A"], p), _half_ec(halves["B"], p)
    from compare import offdiag_flat
    out = {}
    for m in ("gc", "gpdc"):
        rho = float(spearmanr(offdiag_flat(ea[m]), offdiag_flat(eb[m])).statistic)
        out[m] = rho
    return out


def ceilings_all(subjects_x: Dict[str, np.ndarray], p: int) -> Dict:
    """subjects_x: {sid: x (C,T)} → {method: {mean, sd, per_subject}}。"""
    per = {}
    for sid, x in subjects_x.items():
        c = ceiling_for_subject(x, p)
        for m, v in c.items():
            per.setdefault(m, {})[sid] = v
    out = {}
    for m, d in per.items():
        arr = np.array(list(d.values()))
        out[m] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                  "per_subject": d}
    return out
