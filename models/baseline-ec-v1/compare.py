"""compare —— 主轨比较器：规范化 → 3782 有向边配对 → 距离偏相关 Spearman 主指标 → 检验。

口径（design §4）：
- 主指标 = 逐被试距离偏相关 Spearman ρ（三向量秩变换后偏相关；距离=通道三维欧氏）；
  未偏相关 ρ 仅作上下文行。
- 检验 = 单样本 t（n=S）× Holm（跨参考法）+ 1000×受试者 bootstrap 95% CI（seed 2026）
  + ρ/split-half 天花板比值。
- 参考值统一为原始边值（用户指定；GC=ln SSR 比、gPDC=|gPDC|² 均非负）。
- EC_pert 接口：npz 键 ec (62,62)[或 (S,62,62)]、readout∈{slope,auc,rstar}、
  subject_ids（list 或 'global'）；契约=大=强（rstar 须调用方先翻转）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import rankdata, ttest_1samp

import config as C


# ---------- 基元 ----------

def offdiag_flat(M: np.ndarray) -> np.ndarray:
    """有向边展平（行=源，行优先序；非三角）。62×62 → 3782。"""
    M = np.asarray(M)
    return np.ravel(M[~np.eye(M.shape[0], dtype=bool)])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def partial_spearman(a: np.ndarray, b: np.ndarray, cvec: np.ndarray) -> float:
    """rank(a),rank(b) 偏掉 rank(c) 后的偏相关（并列取平均秩）。"""
    ra, rb, rc = (rankdata(v) for v in (a, b, cvec))
    r_ab, r_ac, r_bc = _pearson(ra, rb), _pearson(ra, rc), _pearson(rb, rc)
    den = np.sqrt(max((1 - r_ac ** 2) * (1 - r_bc ** 2), 1e-300))
    return float((r_ab - r_ac * r_bc) / den)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni 校正 p（升序乘 (m−rank+1) 后单调化）。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for rank, (k, p) in enumerate(items, start=1):
        prev = max(prev, min(1.0, p * (m - rank + 1)))
        adj[k] = float(prev)
    return adj


def bootstrap_ci_mean(vals: np.ndarray, n_boot: int = C.BOOTSTRAP_N,
                      seed: int = C.BOOTSTRAP_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    S = len(vals)
    idx = rng.integers(0, S, size=(n_boot, S))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- EC_pert 接口 ----------

def load_ec_pert(path) -> Dict:
    d = np.load(path, allow_pickle=True)
    ec = np.asarray(d["ec"], dtype=np.float64)
    readout = str(d["readout"])
    assert readout in ("slope", "auc", "rstar"), f"未知 readout {readout!r}"
    if "orientation" in d:
        if str(d["orientation"]) != "larger_is_stronger":
            raise ValueError(
                "EC_pert 契约：大=强（rstar 须调用方先翻转，如取 −r* 或 1/r*）")
    out = {"readout": readout, "subject_ids": d["subject_ids"] if "subject_ids" in d else "global"}
    if np.any(ec < 0):
        if readout == "rstar":
            raise ValueError("rstar readout 出现负值：方向未翻转（契约=大=强）")
        out["ec_sign"] = np.sign(ec)          # slope/auc 异常负值：存档符号、主轨取 |·|
        ec = np.abs(ec)
    out["ec"] = ec
    return out


# ---------- 主比较 ----------

def run_comparison(refs: Dict[str, np.ndarray],           # {method: (S,62,62)}
                   subject_ids, ec_pert: Dict,
                   dist: np.ndarray,                       # (62,62) 通道距离
                   ceilings: Optional[Dict[str, float]] = None) -> Dict:
    S = next(iter(refs.values())).shape[0]
    dvec = offdiag_flat(dist)
    ec = ec_pert["ec"]
    per_subj = "global" if ec.ndim == 2 else list(ec_pert.get("subject_ids", subject_ids))
    results = {"n_subjects": S, "readout": ec_pert["readout"],
               "ec_pert_granularity": "global" if ec.ndim == 2 else "per_subject",
               "methods": {}}
    rhos_partial, rhos_raw = {}, {}
    for m, mats in refs.items():
        assert mats.shape == (S, C.N_CHANNELS, C.N_CHANNELS)
        rp, rr = np.zeros(S), np.zeros(S)
        for s in range(S):
            v_p = offdiag_flat(ec if ec.ndim == 2 else ec[s])
            v_m = offdiag_flat(mats[s])
            rp[s] = partial_spearman(v_p, v_m, dvec)
            rr[s] = _pearson(rankdata(v_p), rankdata(v_m))   # 未偏相关上下文（=Spearman）
        t, p = ttest_1samp(rp, 0.0)
        rhos_partial[m], rhos_raw[m] = rp, rr
        entry = {
            "rho_partial_mean": float(rp.mean()),
            "rho_partial_sd": float(rp.std(ddof=1)),
            "rho_partial_t": float(t), "rho_partial_p": float(p),
            "rho_partial_ci95": bootstrap_ci_mean(rp),
            "rho_raw_mean": float(rr.mean()),
            "rho_raw_ci95": bootstrap_ci_mean(rr),
            "rho_partial_per_subject": rp.round(6).tolist(),
        }
        if ceilings and m in ceilings:
            entry["ceiling"] = ceilings[m]
            entry["ceiling_ratio"] = float(rp.mean() / ceilings[m]) if ceilings[m] > 0 else None
        results["methods"][m] = entry
    results["holm_p"] = holm({m: results["methods"][m]["rho_partial_p"] for m in refs})
    return results


def synthetic_ec_pert_demo(ref_mean: np.ndarray, seed: int = 2026,
                            noise: float = 0.3) -> np.ndarray:
    """比较器联调用合成 EC_pert（非科学产物，json 中标记 synthetic）。"""
    rng = np.random.default_rng(seed)
    z = (ref_mean - ref_mean.min()) / (ref_mean.max() - ref_mean.min() + 1e-12)
    pert = np.clip(z + noise * rng.standard_normal(z.shape), 0.0, None)
    np.fill_diagonal(pert, 0.0)
    return pert
