"""report.py — S/G 矩阵 + refs 配对相关四件套 + 距离偏 + 阴性对照（design.md §4 判读口径）。

判读口径（以 ni-mse-design v1.5 §1.1 为准；design.md §4 实例化）：
1. 主口径 = 逐被试配对：EC_pert_s（S 矩阵，行=源/列=靶）× ec_refs_s（offdiag 3,782 对
   直接配对，无转置）→ Pearson + Spearman；n=10 汇报 mean±std / median / range；
2. 副口径 = 组级单次（10 例 S 逐元素中位 vs 同 10 例 refs 均值）；48 例 refs 全均值敏感性；
3. flip 检查 = 同式对 ec.T（direct ≈ flipped ⇒ 相关由对称分量承载，方向信息弱）；
4. 距离偏判定轨 = 双方对 pairwise_d[off] OLS 残差化后再 Spearman（组级 + 逐被试随附）；
   pairwise_d 来自 θ* cfg.geometry_path（fmt 原生 geometry_edra_fmt.pt，只读）；
5. 阴性对照 = c=0.05 臂 G 通过率（Q90 名义 ~10%）与 S 分布位置（→1 附近）；
6. R5（被试级离群旗标）/ R6（S>10 超敏边旗标）/ R3（split-half >10%）/ Q95 旗标带随附。

refs 只读：`refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/
refs_s{sid}.npz` 的 `ec` 键（(62,62)，行=源；对角 0）。纯 CPU 矩阵运算（scipy.stats）。
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先

import datetime
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

import thostar as TH

REFS_DIR = Path("/root/npi/refs/granger_bivariate_spectral/"
                "edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65")
C_MAIN = 0.5          # 主读出臂
C_NEG = 0.05          # 阴性对照臂
C_SENS = (0.1, 0.3)   # 敏感性臂
NOMINAL_Q90 = 0.10    # 阴性对照名义通过率

_OFF = ~np.eye(62, dtype=bool)
N_OFFDIAG = 3782


def load_refs_ec(sid: int) -> np.ndarray:
    """refs_s{sid}.npz 的 ec 键（只读；(62,62) 行=源、对角 0、offdiag ≥ 0）。"""
    z = np.load(REFS_DIR / f"refs_s{sid}.npz")
    ec = np.asarray(z["ec"], dtype=np.float64)
    assert ec.shape == (62, 62), f"refs_s{sid} ec 形状 {ec.shape} != (62,62)"
    return ec


def refs_all_ids() -> list:
    import re
    ids = []
    for f in REFS_DIR.glob("refs_s*.npz"):
        m = re.fullmatch(r"refs_s(\d+)\.npz", f.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_pairwise_d() -> np.ndarray:
    """通道对欧氏距离（geometry_edra_fmt.pt coords cdist；路径取自装载 cfg，禁字面量）。"""
    import torch
    cfg = TH.load_cfg()
    g = torch.load(cfg.geometry_path, map_location="cpu", weights_only=False)
    coords = torch.as_tensor(g["coords"], dtype=torch.float64)
    return torch.cdist(coords, coords).numpy()


def ols_residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS y ~ [1, x] 残差（距离偏判定轨：双方残差化后再相关）。"""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def pair_statistics(S: np.ndarray, ec: np.ndarray, d: np.ndarray) -> dict:
    """单对 (S, ec) 的相关四件套（offdiag 展平；direct 主 / flip 副 / 距离偏）。"""
    off = _OFF
    s, e, dd = S[off], ec[off], d[off]
    out = {
        "pearson_direct": float(sps.pearsonr(s, e).statistic),
        "spearman_direct": float(sps.spearmanr(s, e).statistic),
        "pearson_flip": float(sps.pearsonr(s, ec.T[off]).statistic),
        "spearman_flip": float(sps.spearmanr(s, ec.T[off]).statistic),
    }
    s_res = ols_residualize(s, dd)
    e_res = ols_residualize(e, dd)
    out["spearman_dist_partial"] = float(sps.spearmanr(s_res, e_res).statistic)
    return out


def _agg(values: list) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "n": int(len(a))}


def report_stage(out_root: Path, p, subjects) -> dict:
    """report 期：前置断言上游产物齐全有限 → S/G + 判读四件套 → report/report.json(+npz)。"""
    null_dir, pert_dir = out_root / "null", out_root / "perturb"
    rep_dir = out_root / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    c_arms = [float(c) for c in p.c_arms]
    # 前置断言：主读出臂必须在场；阴性对照/敏感性臂缺场（冒烟缩参口径，Run 001 实测暴露：
    # 原版强制四臂断言使 run.py 登记的单臂冒烟 report 期必挂）⇒ 对应节段显式登记「缺场」，
    # 不静默——主 run 四臂全在场时各节段数值与原版逐字段一致。
    assert C_MAIN in c_arms, f"report 前置断言：主读出臂 {C_MAIN} 不在 perturb 参数 {c_arms}"
    neg_present = C_NEG in c_arms
    sens_present = {c: (c in c_arms) for c in C_SENS}
    _absent = lambda c: {"skipped": f"c={c:g} 臂不在本次 c_arms {c_arms}（缩参冒烟口径，节段缺场）"}  # noqa: E731
    # ---- 前置断言：10×|c 臂| perturb + 10 null 产物齐全有限（§1b-7）----
    tau_b = {}
    for sid in subjects:
        nz = np.load(null_dir / f"{sid}.npz")
        assert np.isfinite(nz["tau_b"]).all(), f"null/{sid}.npz tau_b 非有限"
        tau_b[sid] = {"tau_b": nz["tau_b"], "tau_q95": nz["tau_q95"],
                      "sh_median": float(nz["sh_median"]), "sh_max": float(nz["sh_max"]),
                      "r3_triggered": bool(nz["r3_triggered"])}
        for c in c_arms:
            pz = np.load(pert_dir / f"{sid}_c{c:g}.npz")
            assert np.isfinite(pz["R_bar"]).all(), f"perturb/{sid}_c{c:g}.npz R_bar 非有限"
    d = load_pairwise_d()

    # ---- S/G 矩阵（逐被试 × c 臂）----
    S = {c: np.stack([np.load(pert_dir / f"{sid}_c{c:g}.npz")["R_bar"].astype(np.float64)
                      / tau_b[sid]["tau_b"] for sid in subjects]) for c in c_arms}
    G = {c: np.stack([np.load(pert_dir / f"{sid}_c{c:g}.npz")["R_bar"].astype(np.float64)
                      > tau_b[sid]["tau_b"] for sid in subjects]) for c in c_arms}
    G_q95 = {c: np.stack([np.load(pert_dir / f"{sid}_c{c:g}.npz")["R_bar"].astype(np.float64)
                          > tau_b[sid]["tau_q95"] for sid in subjects]) for c in c_arms}

    # ---- ① 主口径：逐被试配对（main c）----
    per_subject = {}
    for i, sid in enumerate(subjects):
        ec = load_refs_ec(sid)
        st = pair_statistics(S[C_MAIN][i], ec, d)
        st["g_pass_rate"] = float(G[C_MAIN][i][_OFF].mean())
        st["g_q95_share"] = float(G_q95[C_MAIN][i][_OFF].mean())
        st["s_offdiag_median"] = float(np.median(S[C_MAIN][i][_OFF]))
        per_subject[str(sid)] = st
    agg_keys = ["pearson_direct", "spearman_direct", "pearson_flip", "spearman_flip",
                "spearman_dist_partial", "g_pass_rate", "s_offdiag_median"]
    subject_aggregates = {k: _agg([per_subject[str(sid)][k] for sid in subjects])
                          for k in agg_keys}

    # ---- ② 副口径：组级单次 + 48 例敏感性 ----
    S_group = np.median(S[C_MAIN], axis=0)
    ec_stack_10 = np.stack([load_refs_ec(sid) for sid in subjects])
    ec_mean_10 = ec_stack_10.mean(axis=0)
    all_ids = refs_all_ids()
    ec_mean_48 = np.stack([load_refs_ec(s) for s in all_ids]).mean(axis=0)
    group_level = {"vs_refs_mean_10": pair_statistics(S_group, ec_mean_10, d),
                   "vs_refs_mean_48": pair_statistics(S_group, ec_mean_48, d),
                   "refs_n_10": len(subjects), "refs_n_48": len(all_ids)}

    # ---- ⑤ 阴性对照（c=0.05）：G 通过率 vs Q90 名义 10%；S 分布 → 1 附近 ----
    if neg_present:
        neg = {
            "g_pass_rate": _agg([float(G[C_NEG][i][_OFF].mean()) for i in range(len(subjects))]),
            "s_offdiag_median": _agg([float(np.median(S[C_NEG][i][_OFF]))
                                      for i in range(len(subjects))]),
            "nominal": NOMINAL_Q90,
        }
    else:
        neg = _absent(C_NEG)
    # ---- 敏感性臂 {0.1, 0.3} ----
    sensitivity = {}
    for c in C_SENS:
        if not sens_present[c]:
            sensitivity[f"{c:g}"] = _absent(c)
            continue
        sensitivity[f"{c:g}"] = {
            "spearman_direct": _agg([pair_statistics(S[c][i], load_refs_ec(sid), d)["spearman_direct"]
                                     for i, sid in enumerate(subjects)]),
            "g_pass_rate": _agg([float(G[c][i][_OFF].mean()) for i in range(len(subjects))]),
        }

    # ---- R5/R6/R3/Q95 旗标 ----
    med_s = np.array([np.median(S[C_MAIN][i][_OFF]) for i in range(len(subjects))])
    med_group = float(np.median(med_s))
    mad = float(np.median(np.abs(med_s - med_group)))
    r5_flags = [int(subjects[i]) for i in range(len(subjects))
                if mad > 0 and abs(med_s[i] - med_group) > 3 * mad]
    r6_count = int((S[C_MAIN] > 10).sum())
    r6_examples = []
    if r6_count:
        idx = np.argwhere(S[C_MAIN] > 10)
        r6_examples = [{"subject": int(subjects[i]), "a": int(a), "b": int(b),
                        "S": float(S[C_MAIN][i, a, b])}
                       for i, a, b in idx[:100]]
    flags = {
        "r3_splithalf": {str(sid): {"sh_median": tau_b[sid]["sh_median"],
                                     "sh_max": tau_b[sid]["sh_max"],
                                     "triggered": tau_b[sid]["r3_triggered"]}
                         for sid in subjects},
        "r3_any_triggered": bool(any(tau_b[sid]["r3_triggered"] for sid in subjects)),
        "r5_subject_outliers": r5_flags,
        "r6_hypersensitive_edges": {"count": r6_count, "examples_capped_100": r6_examples},
        "q95_share_main": _agg([float(G_q95[C_MAIN][i][_OFF].mean())
                                for i in range(len(subjects))]),
    }

    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_root": str(out_root),
        "subjects": [int(s) for s in subjects],
        "main_c": C_MAIN, "negative_control_c": C_NEG, "sensitivity_c": list(C_SENS),
        "offdiag_pairs": N_OFFDIAG,
        "orientation": "S[a,b]: 行=源 a / 列=靶 b；refs ec 同向（direct 无转置）",
        "per_subject_main_c": per_subject,
        "subject_aggregates_main_c": subject_aggregates,
        "group_level_secondary": group_level,
        "negative_control": neg,
        "sensitivity": sensitivity,
        "flags": flags,
        "caliber_notes": [
            "主口径=逐被试配对 n=10（v1.5 §1.1）；组级单次为副口径",
            "flip≈direct ⇒ 相关由对称分量承载，方向信息弱（判读非门槛）",
            "距离偏 Spearman = 双方对 pairwise_d OLS 残差化后再相关（判读锚点）",
            "不对生成信号重算 GC（v1.4 拍板）；refs 只读复用",
        ],
    }
    (rep_dir / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez(rep_dir / "report_matrices.npz",
             subjects=np.asarray(subjects, dtype=np.int64),
             c_arms=np.asarray(c_arms, dtype=np.float64),
             S=np.stack([S[c] for c in c_arms]).astype(np.float32),      # (n_c, n_sub, 62, 62)
             G_main=G[C_MAIN].astype(bool), G_q95_main=G_q95[C_MAIN].astype(bool),
             tau_b=np.stack([tau_b[sid]["tau_b"] for sid in subjects]).astype(np.float32),
             tau_q95=np.stack([tau_b[sid]["tau_q95"] for sid in subjects]).astype(np.float32),
             S_group=S_group.astype(np.float32))
    return report
