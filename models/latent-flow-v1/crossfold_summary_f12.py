#!/usr/bin/env python3
"""crossfold_summary_f12.py — 跨折（fold0/1/2）symw EC 三折对照汇总（任务 10-25-symw-ec-f12 Run 005）.

用户核心交付③（2026-10-25）：f12 双折 EC 结果与 fold0 可直接汇总比较。本脚本
**读档汇总**（纯后处理，零 GPU/零随机源/零重计算）：
  fold0 = outputs/10-25-symw-ec-unbiased/{summary/group_summary.json,
          recompute_v4/gccorr_dual_summary.json}（在档任务 10-25-symw-ec-unbiased，
          **只读引用**）
  fold1/2 = outputs/10-25-symw-ec-f12/fold{k}/{summary/group_summary.json,
          recompute_v4/gccorr_dual_summary.json}（本任务 Run 001–004 产物）

三折同口径（逐字，唯一实质差异 = 各折代理脑 ckpt 与 test 被试集）：无偏 μ=0
高斯 σ·ξ 扰动、global-τ q=0.5@σ_ref=0.3、σ 网格 log[0.01,0.48]×24、seed 42、
bf16 autocast、每折 test 8 例 × 30 trial 全量、τ 分位含对角；conn 交付公式
v4 = −(ln σ*+0.5·ln2πe)+5/7（各折重映射/全正/abs≡raw 断言 8/8 + 参考 Pearson
平移不变；在档 dual 档为双参考时代产物 asserts 40/折，2026-10-27 线配置固定
取消 v2 对比项后 v1 单参考复跑为 32/折，两态均受——见 load_fold 断言）。

**判读边界（随档 caliber_note）**：三折各用不同 θ 实例（各折独立训练的 symw
代理脑）+ 不同 test 被试集（5-fold manifest 互斥三分支）⇒ 跨折并置为
DESCRIPTIVE（描述性并置），数字差异 = 代理脑实例差异 + 被试差异 + 噪声，
**不作复现性判读**；单 seed（42）。EC 结论继承 symw 代理脑 provisional 状态
（θ 候选判断待用户）。

conn 分布行口径注记：group_summary 的 conn_mean/std_offdiag = 房内 v1 公式
−(ln σ*+ln2πe)+1（summary 源档）；v4 = v1 + 0.5·ln2πe − 2/7（+1.1332）严格
常数平移 ⇒ 跨折行内比较（差值/离散度）不受公式选择影响，均值绝对值差 1.1332。

Run:
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/crossfold_summary_f12.py
"""
from __future__ import annotations

import os

# --- BLAS 守卫：numpy 导入前（npi 环境契约）---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TASK_ROOT = "/root/npi/outputs/10-25-symw-ec-f12"
DEFAULT_OUT = f"{TASK_ROOT}/crossfold/crossfold_summary.json"

# 三折登记（folds registry）：ckpt/best-val = 各折训练侧登记（PRD 关键事实 +
# fold0 README）；来源任务 = 各折 EC 档案归属。fold0 树只读。
FOLDS = {
    0: {
        "label": "fold0",
        "source_task": "10-25-symw-ec-unbiased (Run 002 summary / Run 003 recompute_v4)",
        "surrogate_task": "10-25-latent-flow-exit-gnn-symw run_symw_fold0",
        "ckpt": ("/root/npi/outputs/10-25-latent-flow-exit-gnn-symw/"
                 "run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt"),
        "best_val": "0.8054@E144 (early-stop converged)",
        "summary_json": ("/root/npi/outputs/10-25-symw-ec-unbiased/summary/"
                         "group_summary.json"),
        "dual_json": ("/root/npi/outputs/10-25-symw-ec-unbiased/recompute_v4/"
                      "gccorr_dual_summary.json"),
        "task_prefix": "10-25-symw-ec-unbiased",
    },
    1: {
        "label": "fold1",
        "source_task": "10-25-symw-ec-f12 (Run 001 ec_main / Run 003 summary / Run 004 recompute_v4)",
        "surrogate_task": "10-25-latent-flow-symw-f12 run_symw_fold1",
        "ckpt": ("/root/npi/outputs/10-25-latent-flow-symw-f12/run_symw_fold1/"
                 "checkpoints/fold1_mild_flow_best_graph_mexhat.pt"),
        "best_val": "0.8264@E116 (early-stop E146, converged)",
        "summary_json": f"{TASK_ROOT}/fold1/summary/group_summary.json",
        "dual_json": f"{TASK_ROOT}/fold1/recompute_v4/gccorr_dual_summary.json",
        "task_prefix": "10-25-symw-ec-f12-fold1",
    },
    2: {
        "label": "fold2",
        "source_task": "10-25-symw-ec-f12 (Run 002 ec_main / Run 003 summary / Run 004 recompute_v4)",
        "surrogate_task": "10-25-latent-flow-symw-f12 run_symw_fold2",
        "ckpt": ("/root/npi/outputs/10-25-latent-flow-symw-f12/run_symw_fold2/"
                 "checkpoints/fold2_mild_flow_best_graph_mexhat.pt"),
        "best_val": "0.8433@E113 (early-stop E143, converged)",
        "summary_json": f"{TASK_ROOT}/fold2/summary/group_summary.json",
        "dual_json": f"{TASK_ROOT}/fold2/recompute_v4/gccorr_dual_summary.json",
        "task_prefix": "10-25-symw-ec-f12-fold2",
    },
}

# 2026-10-27 线配置固定：v2_1_125_aic4_25 对比项取消（在档 dual json 的 v2 块
# 不再汇总，refs 物理文件已删）
REFS_KEYS = ("v1_1_45_fixed8",)

CALIBER_NOTE = (
    "三折同口径逐字（唯一实质差异 = 各折代理脑 ckpt 与 test 被试集）：无偏 mu=0 "
    "高斯 sigma*xi 扰动（z-score 后、入模前 x_obs(375) 源通道）、global-tau "
    "q=0.5@sigma_ref=0.3、sigma 网格 log[0.01,0.48]x24、seed 42、bf16 autocast、"
    "每折 test 8 例 x 30 trial 全量、tau 分位含对角自响应；conn 交付 v4 = "
    "-(ln sigma*+0.5*ln2pie)+5/7（断言组：重映射单调/全正/abs==raw 8/8 + 参考 "
    "Pearson 平移不变；在档 dual 档 40/折、2026-10-27 取消 v2 对比项后单参考"
    "复跑 32/折，两态均受）。**跨折并置为 DESCRIPTIVE（描述性口径）**："
    "各折 theta 为不同代理脑实例（独立训练）、test 被试集互斥（5-fold manifest "
    "三分支）⇒ 数字差异 = 代理脑实例差异 + 被试差异 + 噪声，不作复现性判读；"
    "单 seed（42）。conn_mean/std_offdiag 行 = 房内 v1 公式口径（summary 源档），"
    "v4 = v1 + 1.1332 严格平移 ⇒ 跨折行内比较不受影响。EC 结论继承 symw 代理脑 "
    "provisional 状态（theta 候选判断待用户，随其联动）。")

DESRIPTIVE_ONLY = (
    "描述性并置登记（先例协议）：EC=代理脑内部量 vs GC=观测数据量，实现口径隔离；"
    "不构成代理脑资格判定；跨折数字不作复现性判读（不同 theta 实例 + 不同被试）。")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    return p.parse_args(argv)


def _agg(block, key):
    return {"mean": block[key]["mean"], "std": block[key]["std"],
            "min": block[key]["min"], "max": block[key]["max"]}


def load_fold(k: int) -> dict:
    reg = FOLDS[k]
    gs = json.load(open(reg["summary_json"]))
    du = json.load(open(reg["dual_json"]))
    # 完整性断言（读档守卫：折身份/规模/断言计数对不上即拒汇总）
    assert gs["task"].startswith(reg["task_prefix"]), \
        f"fold{k} summary task 串不符: {gs['task']}"
    assert du["task"].startswith(reg["task_prefix"]), \
        f"fold{k} dual task 串不符: {du['task']}"
    assert gs["n_subjects"] == 8, f"fold{k} n_subjects != 8"
    # 断言计数两态：40 = 双参考时代在档档（v1+v2 各 8 例 Pearson 平移检查）；
    # 32 = 2026-10-27 线配置固定（v2 对比项取消）后 v1 单参考复跑档；其余值 = 档身份异常仍拒
    assert du["asserts_passed"] in (40, 32), \
        f"fold{k} asserts {du['asserts_passed']} not in (40 dual-era, 32 v1-only)"
    remap = [p["spearman_remap_invariance"] for p in gs["per_subject"]]
    assert remap == ["PASS"] * 8, f"fold{k} 重映射不变性非 8/8 PASS"
    subjects = sorted(p["subject"] for p in gs["per_subject"])
    assert subjects == sorted(set(subjects)) and len(subjects) == 8
    return {"registry": reg, "gs": gs, "du": du}


def fold_block(k: int, loaded: dict) -> dict:
    reg, gs, du = loaded["registry"], loaded["gs"], loaded["du"]
    g, vd = gs["group"], gs["var_decomp_group_mean"]
    taus = [p["tau"] for p in gs["per_subject"]]
    gc = {}
    for rk in REFS_KEYS:
        b = du["refs_blocks"][rk]
        gc[rk] = {
            "group_spearman_raw_aligned": _agg(b["group"], "spearman_raw_aligned"),
            "group_pearson_raw_aligned": _agg(b["group"], "pearson_raw_aligned"),
            "group_spearman_raw_flipped": _agg(b["group"], "spearman_raw_flipped"),
            "group_mean_spearman_raw_aligned":
                b["group_mean"]["spearman_raw_aligned"],
            "group_mean_pearson_raw_aligned":
                b["group_mean"]["pearson_raw_aligned"],
        }
    return {
        "source_task": reg["source_task"],
        "surrogate_task": reg["surrogate_task"],
        "ckpt": reg["ckpt"],
        "best_val": reg["best_val"],
        "n_subjects": gs["n_subjects"],
        "test_subjects": [p["subject"] for p in gs["per_subject"]],
        "interior_pct": _agg(g, "pct_interior"),
        "pct_at_min": _agg(g, "pct_at_min"),
        "pct_at_max": _agg(g, "pct_at_max"),
        "p_response_above_tau": _agg(g, "p_response_above_tau"),
        "dyn_ratio_median": _agg(g, "dyn_ratio_median"),
        "tau_median_across_subjects": float(np.median(taus)),
        "tau_min": float(min(taus)), "tau_max": float(max(taus)),
        "sigma_median_offdiag": _agg(g, "sigma_median_offdiag"),
        "conn_mean_offdiag_v1_caliber": _agg(g, "conn_mean_offdiag"),
        "conn_std_offdiag_v1_caliber": _agg(g, "conn_std_offdiag"),
        "var_decomp_group_mean": {
            "row_source_share": vd["row_source_share"],
            "col_target_share": vd["col_target_share"],
            "residual_edge_share": vd["residual_edge_share"]},
        "cross_subject_stability_mean_offdiag_spearman":
            gs["cross_subject_stability"]["mean_offdiag_spearman"],
        "remap_invariance": "PASS 8/8",
        "gc_dual": gc,
    }


def per_subject_gc_detail(loads: dict) -> dict:
    """逐被试 GC rho 明细（v1 单参考）：24 例（fold0/1/2 各 8）三折全景，含组别与折归属。"""
    out = {rk: {} for rk in REFS_KEYS}
    for k in sorted(loads):
        reg, gs, du = loads[k]["registry"], loads[k]["gs"], loads[k]["du"]
        ec_by_sid = {p["subject"]: p for p in gs["per_subject"]}
        for rk in REFS_KEYS:
            for sid_key, r in du["refs_blocks"][rk]["per_subject"].items():
                sid = r["subject"]
                ec = ec_by_sid[sid]
                out[rk][sid] = {
                    "fold": k, "group": r["group"],
                    "spearman_raw_aligned": r["spearman_raw_aligned"],
                    "pearson_raw_aligned": r["pearson_raw_aligned"],
                    "spearman_raw_flipped": r["spearman_raw_flipped"],
                    "ec_context": {"tau": ec["tau"],
                                   "pct_interior": ec["pct_interior"],
                                   "conn_mean_offdiag_v1_caliber":
                                       ec["conn_mean_offdiag"],
                                   "dyn_ratio_median": ec["dyn_ratio_median"]},
                }
    return out


def main() -> int:
    a = parse_args()
    loads = {k: load_fold(k) for k in FOLDS}
    blocks = {FOLDS[k]["label"]: fold_block(k, loads[k]) for k in sorted(FOLDS)}
    detail = per_subject_gc_detail(loads)
    for rk in REFS_KEYS:
        assert len(detail[rk]) == 24, \
            f"{rk} 逐被试明细非 24 例（三折并集应无重叠）: {len(detail[rk])}"

    summary = {
        "task": ("10-25-symw-ec-f12 Run 005 (cross-fold summary fold0/1/2, "
                 "read-only aggregation of archived group summaries + GC "
                 "blocks [v1-only summarized since 2026-10-27 config fix])"),
        "caliber_note": CALIBER_NOTE,
        "descriptive_only": DESRIPTIVE_ONLY,
        "model_version": ("FlowVelocityNetExitGNN（K=2 共享 A_eff 消息传递 + "
                          "zero-init exit projection，1,276,258 参数）+ SymWAdam "
                          "每步 W←(W+Wᵀ)/2 投影 + anti-uniform λ0.2；symw 训练 "
                          "Run 001 同参跨折（10-25-latent-flow-exit-gnn-symw "
                          "fold0 / 10-25-latent-flow-symw-f12 fold1/2）；λ/sym-W "
                          "两因素同变未消融"),
        "refs": {
            "v1_1_45_fixed8": ("mne-connectivity spectral gc, VAR order 8 fixed, "
                               "1-45 Hz faverage (refs mild40-trial500-zscore-"
                               "fs250-fixed8-f1-45-favg)"),
            "protocol": ("offdiag 16256 pairs; aligned = row=source vs ec; "
                         "flipped = vs ec.T; Spearman 主读数 + Pearson 照登；"
                         "abs == raw（v4 全正，逐位断言）⇒ abs/raw 块逐位相同；"
                         "2026-10-27 线配置固定：v2 1-125hz 对比项已取消"),
        },
        "provenance": {
            FOLDS[k]["label"]: {
                "ckpt": FOLDS[k]["ckpt"], "best_val": FOLDS[k]["best_val"],
                "surrogate_task": FOLDS[k]["surrogate_task"],
                "ec_source_task": FOLDS[k]["source_task"],
                "summary_json": FOLDS[k]["summary_json"],
                "dual_json": FOLDS[k]["dual_json"],
                "read_only": k == 0,
            } for k in sorted(FOLDS)
        },
        "per_fold": blocks,
        "per_subject_gc_detail": detail,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    # 控制台三折对照小表（详细表落 README）
    cols = ["fold0", "fold1", "fold2"]
    rows = [
        ("best_val", lambda b: b["best_val"].split(" ")[0]),
        ("interior%", lambda b: f"{b['interior_pct']['mean']:.1f}±{b['interior_pct']['std']:.1f}"),
        ("dyn_ratio", lambda b: f"{b['dyn_ratio_median']['mean']:.2f}x"),
        ("var row/col/resid", lambda b: (
            f"{b['var_decomp_group_mean']['row_source_share']:.3f}/"
            f"{b['var_decomp_group_mean']['col_target_share']:.3f}/"
            f"{b['var_decomp_group_mean']['residual_edge_share']:.3f}")),
        ("stab rho", lambda b: f"{b['cross_subject_stability_mean_offdiag_spearman']:.3f}"),
        ("GC rho (v1)", lambda b: (
            f"{b['gc_dual']['v1_1_45_fixed8']['group_spearman_raw_aligned']['mean']:+.4f}"
            f"±{b['gc_dual']['v1_1_45_fixed8']['group_spearman_raw_aligned']['std']:.4f}")),
    ]
    print(f"[crossfold] {'':16s} " + " | ".join(f"{c:>22s}" for c in cols))
    for name, fn in rows:
        print(f"[crossfold] {name:16s} " +
              " | ".join(f"{fn(blocks[c]):>22s}" for c in cols))
    print(f"[crossfold] per-subject GC detail: 24 例 x 1 refs (v1) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
