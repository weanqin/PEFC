#!/usr/bin/env python3
"""compare_rev —— refs-01（正向）vs rev-01（时间反转）对比交付（用户 2026-10-18 追加）。

读取两端已落盘 refs_all.npz（refs-01 正向 / rev-01 含 reversed=True 标记）**纯计算**：
不重跑拟合、无 RNG、summary 不含时间戳/计时字段——确定性契约（重跑输出逐位一致）。

交付（全部落 OUTPUTS_ROOT/<rev-run>/，refs-01 与 rev-01 既有产物只读不覆写）：
  compare_summary.json                       组级 + Top-K 重叠 + 方向翻转 + 通道级 + 逐例交叉验证
  figs/compare_group_means_fwd_rev_revT.png  ec_matrix 三联 GC̄_fwd | GC̄_rev | GC̄_revᵀ（既有图型复用）
  figs/compare_edges_fwd_vs_rev_revT.png     edge_scatter 边级散点 + 恒等线（visualizer 2026-10-18 增量图型）

对比指标（design §2.4 预登记判读框架的组级延伸，描述性、非硬门限）：
  组级      GC̄_fwd/GC̄_rev 组均值矩阵，3782 有向边 Spearman ρ + Pearson r（同位 vs 镜像位 revᵀ）
  Top-K     fwd Top-50 边 vs rev / revᵀ Top-50 交集数与 Jaccard
  方向翻转  fwd Top-50 边 (A→B) 在 rev 中的同位 vs 镜像位取值均值
  通道级    出强度（offdiag 行和）/入强度（列和）fwd-rev Spearman（出-出 / 出-入 / 入-入）
  逐例      ρ(fwd_s, rev_s) 重算 vs rev-01 refs_summary 已登记 edge_fwd_rev（allclose 交叉验证断言）

用法：python3 models/bivar-gc-ec-v2/compare_rev.py [--fwd-run refs-01] [--rev-run rev-01] [--top-k 50]
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md 惯例：必须在 numpy 之前；setdefault 允许外部覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse           # noqa: E402
import json               # noqa: E402
import sys                # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np        # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

# ---- sys.path 装配（run.py 同式）：v2 目录居首（data_io/bgec2_config 解析到 v2）----
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data_io            # noqa: E402（v2 版；连带经唯一名配方装载 bgec2_config）
import bgec2_config as C  # noqa: E402（v2 config，唯一名）

# ---- 逐例交叉验证容差：登记值以 f64 计算态比较、本交付以 f32 落盘态重算，----
# ---- 偏差属 f32 量化带（实测 max|Δ|≈1.7e-7）                               ----
_CV_RTOL = 1e-6
_CV_ATOL = 1e-6


def _load_refs(path: Path, *, expect_reversed: bool):
    """refs_all.npz → (ec f64 (48,62,62), sids)；装载断言沿 data_io 通用断言同式。

    断言：形状 (n,62,62) / subject_ids == EXPECTED_IDS（排序堆叠契约）/ reversed
    标记按端校验 / 全有限 / 对角=0 / offdiag≥0（实现正确性契约，落盘态复核）。
    """
    d = np.load(path, allow_pickle=True)
    if expect_reversed:
        assert "reversed" in d and bool(d["reversed"]), f"{path}: 缺 reversed=true 标记"
    else:
        assert "reversed" not in d, f"{path}: 意外含 reversed 标记（fwd 端应为正向产物）"
    ec = np.asarray(d["bivar_gc"], dtype=np.float64)
    sids = [str(s) for s in d["subject_ids"]]
    n = ec.shape[0]
    assert ec.shape == (n, C.N_CHANNELS, C.N_CHANNELS), f"{path}: 形状 {ec.shape} 异常"
    assert sids == [str(s) for s in C.EXPECTED_IDS], f"{path}: subject_ids 与 EXPECTED_IDS 不一致"
    eye = ~np.eye(C.N_CHANNELS, dtype=bool)
    assert np.isfinite(ec).all(), f"{path}: 非有限值"
    assert float(np.abs(np.diagonal(ec, axis1=1, axis2=2)).max()) == 0.0, f"{path}: 对角非零"
    assert (ec[:, eye] >= 0).all(), f"{path}: offdiag GC < 0（落盘态复核失败）"
    return ec, sids


def _edge_corr(x: np.ndarray, y: np.ndarray) -> dict:
    """3782 边向量 Spearman ρ + Pearson r（确定性纯计算）。"""
    return {"spearman_rho": float(spearmanr(x, y).statistic),
            "pearson_r": float(pearsonr(x, y).statistic)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fwd-run", default="refs-01", help="正向基准 run 目录名")
    ap.add_argument("--rev-run", default="rev-01", help="时间反转 run 目录名（产物落此处）")
    ap.add_argument("--top-k", type=int, default=50, help="Top-K 边重叠的 K")
    args = ap.parse_args()

    fwd_dir = C.OUTPUTS_ROOT / args.fwd_run
    rev_dir = C.OUTPUTS_ROOT / args.rev_run
    fwd_path, rev_path = fwd_dir / "refs_all.npz", rev_dir / "refs_all.npz"
    assert fwd_path.exists(), f"正向基准缺失：{fwd_path}"
    assert rev_path.exists(), f"反转产物缺失：{rev_path}——先跑 --stage reverse --run-id {args.rev_run}"

    # ---- 装载 + 通用断言（两端）----
    F, sids_f = _load_refs(fwd_path, expect_reversed=False)
    R, sids_r = _load_refs(rev_path, expect_reversed=True)
    assert sids_f == sids_r, "fwd/rev subject_ids 顺序不一致"

    # ---- 数据契约侧复核：manifest ok 清单 == EXPECTED_IDS，groups 与两端 summary 一致 ----
    manifest = data_io.load_manifest()
    ok_sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert ok_sids == list(C.EXPECTED_IDS), f"manifest ok 清单与 EXPECTED_IDS 不一致：{ok_sids}"
    groups = {str(s): manifest[s]["group"] for s in C.EXPECTED_IDS}
    for run_dir in (fwd_dir, rev_dir):
        sj = json.loads((run_dir / "refs_summary.json").read_text())
        assert sj["groups"] == groups, f"{run_dir}: refs_summary groups 与 manifest 不一致"

    # ---- 展平（行主序 offdiag 3782 有向边；行=源/列=靶）----
    eye = ~np.eye(C.N_CHANNELS, dtype=bool)
    ii, jj = np.where(eye)                       # 与 M[eye] 同序（行主序）
    n_edges = int(eye.sum())
    assert n_edges == C.N_CHANNELS * (C.N_CHANNELS - 1) == 3782
    k = args.top_k

    # ---- 组级：组均值矩阵 GC̄_fwd / GC̄_rev（及其镜像 revᵀ）----
    Gf, Gr = F.mean(axis=0), R.mean(axis=0)
    Grt = Gr.T
    of, orv, orvt = Gf[eye], Gr[eye], Grt[eye]
    edge_corr = {"fwd_vs_rev": _edge_corr(of, orv), "fwd_vs_revT": _edge_corr(of, orvt)}

    # ---- Top-K 重叠（确定性：argsort 降序 stable，并列按索引序）----
    def _top_idx(v: np.ndarray) -> np.ndarray:
        return np.argsort(-v, kind="stable")[:k]

    def _top_set(v: np.ndarray) -> set:
        idx = _top_idx(v)
        return set(zip(ii[idx].tolist(), jj[idx].tolist()))

    set_f, set_r, set_rt = _top_set(of), _top_set(orv), _top_set(orvt)
    topk = {}
    for name, s_b in (("fwd_vs_rev", set_r), ("fwd_vs_revT", set_rt)):
        inter = len(set_f & s_b)
        topk[name] = {"intersection": inter, "jaccard": inter / len(set_f | s_b)}

    # ---- 方向翻转刻画：fwd Top-50 边 (A→B) 在 rev 中的同位 vs 镜像位 ----
    kf = _top_idx(of)
    topk_flip = {"k": k,
                 "fwd_top50_mean": float(of[kf].mean()),
                 "rev_same_position_mean": float(orv[kf].mean()),
                 "rev_mirror_position_mean": float(Gr[jj[kf], ii[kf]].mean())}

    # ---- 通道级：出强度（offdiag 行和）/ 入强度（offdiag 列和；对角=0）----
    combos = {"out_fwd_vs_out_rev": (lambda A, B: (A.sum(1), B.sum(1))),
              "out_fwd_vs_in_rev": (lambda A, B: (A.sum(1), B.sum(0))),
              "in_fwd_vs_in_rev": (lambda A, B: (A.sum(0), B.sum(0)))}
    ch_group = {}
    ch_per = {name: [] for name in combos}
    for name, get in combos.items():
        xg, yg = get(Gf, Gr)
        ch_group[name] = float(spearmanr(xg, yg).statistic)
        for s in range(F.shape[0]):
            xs, ys = get(F[s], R[s])
            ch_per[name].append(float(spearmanr(xs, ys).statistic))

    # ---- 逐例 ρ(fwd_s, rev_s) 重算 + 与已登记 edge_fwd_rev 的交叉验证断言 ----
    rev_summary = json.loads((rev_dir / "refs_summary.json").read_text())
    registered = rev_summary["metrics"]["edge_fwd_rev"]["per_subject"]
    rhos = [float(spearmanr(F[s][eye], R[s][eye]).statistic) for s in range(F.shape[0])]
    diffs = [abs(r - registered[sid]) for r, sid in zip(rhos, sids_f)]
    assert all(abs(r - registered[sid]) <= _CV_ATOL + _CV_RTOL * abs(registered[sid])
               for r, sid in zip(rhos, sids_f)), \
        "逐例 ρ(fwd,rev) 与 rev-01 refs_summary 已登记 edge_fwd_rev 不一致（超出 f32 量化带容差）"
    rho_arr = np.asarray(rhos, dtype=np.float64)

    # ---- compare_summary.json（无时间戳/计时字段——确定性契约）----
    def _grp(vals: list) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}

    summary = {
        "deliverable": "正向 vs 时间反转对比（用户 2026-10-18 追加；读 refs-01/rev-01 "
                       "已落盘产物纯计算，不重跑拟合）",
        "n_subjects": len(sids_f), "subject_ids": sids_f, "groups": groups,
        "inputs": {"fwd": {"run_id": args.fwd_run, "path": str(fwd_path),
                           "shape": list(F.shape)},
                   "rev": {"run_id": args.rev_run, "path": str(rev_path),
                           "shape": list(R.shape), "reversed_flag": True}},
        "caliber": {
            "edge_flattening": "offdiag 行主序展平（62×62 对角剔除；行=源/列=靶；"
                               "3782 有向边）",
            "topk_rule": "np.argsort(-vals, kind='stable') 前 K（确定性；并列按索引序）",
            "top_k": k,
            "interpretation_frame": "描述性指纹（预登记，非硬门限；design §2.4）：二阶平稳"
                                    "高斯过程时间反演二阶结构不变、反向 VAR 系数不同 ⇒ "
                                    "GC_rev 与 GC_fwd 既不相等也不必归零；ρ 与幅值联合判读",
        },
        "group_mean": {
            "offdiag_stats": {
                "fwd": {"min": float(of.min()), "median": float(np.median(of)),
                        "max": float(of.max())},
                "rev": {"min": float(orv.min()), "median": float(np.median(orv)),
                        "max": float(orv.max())},
            },
            "edge_corr": edge_corr,
            "topk_overlap": {"k": k, "n_edges": n_edges, **topk},
            "topk_flip": topk_flip,
        },
        "channel_strength_spearman": {
            "definition": "出强度 = offdiag 行和、入强度 = offdiag 列和（对角=0）；62 通道",
            "group_level": ch_group,
            "per_subject": {name: _grp(vals) for name, vals in ch_per.items()},
        },
        "per_subject_edge_rho": {
            "fwd_vs_rev": {**_grp(rhos), "per_subject": {sid: r for sid, r in zip(sids_f, rhos)}},
            "cross_validation": {
                "target": f"{args.rev_run}/refs_summary.json::metrics.edge_fwd_rev.per_subject",
                "assert": f"np.allclose(rtol={_CV_RTOL}, atol={_CV_ATOL})",
                "passed": True,
                "max_absdiff": float(max(diffs)),
                "note": "登记值以 f64 计算态比较、本交付以 f32 落盘态重算——偏差属 "
                        "f32 量化带（落盘口径 float32 落盘 / float64 计算）",
            },
        },
        "figs": ["figs/compare_group_means_fwd_rev_revT.png",
                 "figs/compare_edges_fwd_vs_rev_revT.png"],
        "command": f"python3 models/bivar-gc-ec-v2/compare_rev.py --fwd-run {args.fwd_run} "
                   f"--rev-run {args.rev_run} --top-k {args.top_k}",
        "entry": "models/bivar-gc-ec-v2/compare_rev.py",
        "determinism": "纯计算无 RNG；无时间戳/计时字段；重跑输出逐位一致",
    }
    out_path = rev_dir / "compare_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    # ---- 落图（common/visualizer 正典路由）----
    _COMMON = _HERE.parent.parent / "common"
    if str(_COMMON) not in sys.path:
        sys.path.append(str(_COMMON))
    from visualizer import Visualizer  # noqa: E402（正典路由）

    viz = Visualizer(str(rev_dir / "figs"))
    n = len(sids_f)
    p1 = viz.render("ec_matrix", {
        "ec_a": Gf, "ec_b": Gr, "ec_c": Grt,
        "label_a": f"fwd mean (n={n})",
        "label_b": "rev mean (time-reversed)",
        "label_c": "revT mean (mirror)",
        "stats": {"rho_fwd_rev": edge_corr["fwd_vs_rev"]["spearman_rho"],
                  "rho_fwd_revT": edge_corr["fwd_vs_revT"]["spearman_rho"]},
    }, "compare_group_means_fwd_rev_revT.png")
    p2 = viz.render("edge_scatter", {
        "panels": [
            {"title": "fwd vs rev (same direction)",
             "x": of, "y": orv,
             "x_label": "GC fwd group mean", "y_label": "GC rev group mean",
             "stat_lines": [f"Spearman rho={edge_corr['fwd_vs_rev']['spearman_rho']:.3f}",
                            f"Pearson r={edge_corr['fwd_vs_rev']['pearson_r']:.3f}",
                            f"n={n_edges} directed edges"]},
            {"title": "fwd vs revT (mirror)",
             "x": of, "y": orvt,
             "x_label": "GC fwd group mean", "y_label": "GC rev$^T$ group mean",
             "stat_lines": [f"Spearman rho={edge_corr['fwd_vs_revT']['spearman_rho']:.3f}",
                            f"Pearson r={edge_corr['fwd_vs_revT']['pearson_r']:.3f}",
                            f"n={n_edges} directed edges"]},
        ],
        "note": "group-mean bivariate GC, offdiag edges; dashed = identity line",
    }, "compare_edges_fwd_vs_rev_revT.png")

    print(f"[compare] summary -> {out_path}")
    print(f"[compare] figs    -> {p1} | {p2}")
    print(f"[compare] 组级: rho(fwd,rev)={edge_corr['fwd_vs_rev']['spearman_rho']:.4f} "
          f"(r={edge_corr['fwd_vs_rev']['pearson_r']:.4f})  "
          f"rho(fwd,revT)={edge_corr['fwd_vs_revT']['spearman_rho']:.4f} "
          f"(r={edge_corr['fwd_vs_revT']['pearson_r']:.4f})")
    print(f"[compare] Top-{k}: fwd∩rev={topk['fwd_vs_rev']['intersection']} "
          f"(J={topk['fwd_vs_rev']['jaccard']:.4f})  "
          f"fwd∩revT={topk['fwd_vs_revT']['intersection']} "
          f"(J={topk['fwd_vs_revT']['jaccard']:.4f})")
    print(f"[compare] 翻转刻画: fwd top{k} 均值={topk_flip['fwd_top50_mean']:.4f} | "
          f"rev 同位={topk_flip['rev_same_position_mean']:.4f} | "
          f"rev 镜像位={topk_flip['rev_mirror_position_mean']:.4f}")
    print(f"[compare] 通道强度: " + "  ".join(f"{n_}={v:.4f}" for n_, v in ch_group.items()))
    print(f"[compare] 逐例交叉验证: max|Δ|={max(diffs):.2e} "
          f"(rtol={_CV_RTOL}, atol={_CV_ATOL}) 全过")


if __name__ == "__main__":
    main()
