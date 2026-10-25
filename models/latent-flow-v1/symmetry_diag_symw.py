#!/usr/bin/env python3
"""symmetry_diag_symw.py — symw 代理脑 EC 方向性对称诊断（任务 10-25-symw-ec-unbiased）.

科学动机（prd R7/R8）：SymWAdam 每步 W←(W+Wᵀ)/2 ⇒ flow 通道耦合严格对称 ⇒
A_eff 无向 ⇒ EC_pert 方向性按构造受限（与有向 GC 参考的相关上限须按无向口径
重设预期）。本脚本对 Run 001 主跑 σ* 做**纯后处理**实证：逐被试 conn 矩阵 M vs
Mᵀ——offdiag 16256 对 Pearson/Spearman + 相对不对称度 ‖M−Mᵀ‖_F/‖M‖_F。

口径注记（比值指标专设）：
  conn v4 = −(ln σ* + 0.5·ln 2πe) + 5/7（主读出，交付公式）
  conn v1 = −(ln σ* + ln 2πe) + 1（房内原式，并置登记）
  v4 = v1 + Δc（Δc = 0.5·ln2πe − 2/7 ≈ +1.1332，严格常数平移）：平移不改变
  ‖M−Mᵀ‖_F（差矩阵中常数相消）但改变 ‖M‖_F ⇒ 两口径不对称度**比值**不同
  （v1 offdiag 均值偏负、v4 全正 ⇒ ‖M_v4‖_F > ‖M_v1‖_F ⇒ v4 比值更小）；
  Pearson/Spearman 对常数平移不变 ⇒ 两口径相关系数数值相同（float64 舍入内）。

输入：ec_main/subject_*.json（Run 001，σ* 矩阵）。输出：
  symmetry_diag/symmetry_diag_summary.json（任务名/口径/公式注记随档）。
纯离线，无 GPU；不改归因算法（只读档重算，σ* float64）。

Run:
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/symmetry_diag_symw.py
"""
from __future__ import annotations

import os

# --- BLAS 守卫：numpy/scipy 导入前（npi 环境契约）---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

TASK = "10-25-symw-ec-unbiased"
DEFAULT_MAIN = "/root/npi/outputs/10-25-symw-ec-unbiased/ec_main"
DEFAULT_OUT = "/root/npi/outputs/10-25-symw-ec-unbiased/symmetry_diag"
HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)
V4_OFFSET = 5.0 / 7.0
CONST_SHIFT = HALF_LOG_2PIE - 2.0 / 7.0  # v4 = v1 + CONST_SHIFT (≈ +1.1332)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--main-dir", type=str, default=DEFAULT_MAIN,
                   help="Run 001 产物目录（subject_*.json）")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT,
                   help="输出目录（symmetry_diag_summary.json）")
    return p.parse_args(argv)


def asym_metrics(m: np.ndarray, off: np.ndarray) -> dict:
    """M vs Mᵀ 对称性三件：offdiag Pearson / Spearman / 相对不对称度."""
    mt = m.T
    diff_norm = float(np.linalg.norm(m - mt))
    base_norm = float(np.linalg.norm(m))
    return {
        "offdiag_pearson_M_vs_MT": float(np.corrcoef(m[off], mt[off])[0, 1]),
        "offdiag_spearman_M_vs_MT": float(sps.spearmanr(m[off], mt[off]).statistic),
        "fro_asym_ratio": diff_norm / base_norm,
        "_fro_diff_norm": diff_norm,  # 内部量：两口径应逐位相等（平移相消）
        "_fro_base_norm": base_norm,
    }


def main() -> int:
    a = parse_args()
    main_dir, out_dir = Path(a.main_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    off = ~np.eye(128, dtype=bool)
    rows = sorted(main_dir.glob("subject_*.json"))
    assert rows, f"no subject_*.json under {main_dir}"

    per_subject = []
    stack_v4, stack_v1 = [], []
    t0 = time.time()
    for f in rows:
        d = json.load(open(f))
        sigma = np.array(d["sigma_matrix"], dtype=np.float64)
        assert sigma.shape == (128, 128), f"{f.name}: {sigma.shape}"
        m_v4 = -(np.log(sigma) + HALF_LOG_2PIE) + V4_OFFSET
        m_v1 = -(np.log(sigma) + 2.0 * HALF_LOG_2PIE) + 1.0
        met_v4 = asym_metrics(m_v4, off)
        met_v1 = asym_metrics(m_v1, off)
        # 口径注记的运行时证据：平移相消 ⇒ 两口径 ‖M−Mᵀ‖ 逐位相等
        assert abs(met_v4["_fro_diff_norm"] - met_v1["_fro_diff_norm"]) < 1e-9, \
            f"{d['subject_file']}: 差矩阵范数口径不一致（平移未相消）"
        per_subject.append({
            "subject": d["subject_file"], "group": d.get("group"),
            "v4": {k: v for k, v in met_v4.items() if not k.startswith("_")},
            "v1": {k: v for k, v in met_v1.items() if not k.startswith("_")},
        })
        stack_v4.append(m_v4)
        stack_v1.append(m_v1)
        print(f"[sym] {d['subject_file']} ({d.get('group')}): "
              f"v4 pearson={met_v4['offdiag_pearson_M_vs_MT']:+.6f} "
              f"spearman={met_v4['offdiag_spearman_M_vs_MT']:+.6f} "
              f"asym={met_v4['fro_asym_ratio']:.6e} "
              f"(v1 asym={met_v1['fro_asym_ratio']:.6e})", flush=True)

    def agg(key, sub_key):
        v = np.array([r[sub_key][key] for r in per_subject], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    group = {sub: {k: agg(k, sub) for k in ("offdiag_pearson_M_vs_MT",
                                            "offdiag_spearman_M_vs_MT",
                                            "fro_asym_ratio")}
             for sub in ("v4", "v1")}
    # 组均值矩阵（跨被试平均后的 M）上的同口径诊断
    gm_v4 = asym_metrics(np.stack(stack_v4).mean(axis=0), off)
    gm_v1 = asym_metrics(np.stack(stack_v1).mean(axis=0), off)

    summary = {
        "task": f"{TASK} (EC directionality symmetry diagnostics, post-hoc)",
        "motivation": (
            "SymWAdam W←(W+Wᵀ)/2 ⇒ A_eff 无向 ⇒ EC 方向性按构造受限；"
            "本诊断为纯后处理实证（读 Run 001 σ* 重算 conn，不改归因算法）"),
        "formulas": {
            "conn_v4": "-(ln sigma* + 0.5*ln(2*pi*e)) + 5/7",
            "conn_v1": "-(ln sigma* + ln(2*pi*e)) + 1",
            "const_shift_v4_vs_v1": float(CONST_SHIFT),
        },
        "caliber_note": (
            "相对不对称度比值口径专设：常数平移（v4=v1+Δc, Δc≈+1.1332）不改变"
            "‖M−Mᵀ‖_F 但改变 ‖M‖_F ⇒ v1/v4 两口径 fro_asym_ratio 数值不同"
            "（v4 为交付主读出，v1 房内口径并置登记）；Pearson/Spearman 对平移"
            "不变 ⇒ 两口径相关系数数值相同（本跑 float64 逐例核过，见 "
            "per_subject）。差矩阵范数两口径逐位相等已断言。"),
        "protocol": ("offdiag 16256 pairs (128x128, diag excluded); "
                     "M vs M.T over same offdiag mask; Pearson + Spearman + "
                     "||M-M.T||_F / ||M||_F"),
        "per_subject": per_subject,
        "group": group,
        "group_mean_matrix": {
            "note": "metrics on the across-subject mean matrix M_bar (v4/v1)",
            "v4": {k: v for k, v in gm_v4.items() if not k.startswith("_")},
            "v1": {k: v for k, v in gm_v1.items() if not k.startswith("_")},
        },
        "reading_anchor": (
            "offdiag corr(M,Mᵀ)→1 且 asym 比值极小 ⇒ EC 方向性按构造坍缩为无向"
            "（与 A_eff 无向 caveat 一致）⇒ 与有向 GC 的相关上限受限，"
            "aligned≈flipped 的方向信息≈零；数值如实登记，判读见 experiment.md"),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = out_dir / "symmetry_diag_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    g4 = group["v4"]
    print(f"\n[sym] GROUP v4: pearson {g4['offdiag_pearson_M_vs_MT']['mean']:+.6f}"
          f"±{g4['offdiag_pearson_M_vs_MT']['std']:.6f} | "
          f"spearman {g4['offdiag_spearman_M_vs_MT']['mean']:+.6f}"
          f"±{g4['offdiag_spearman_M_vs_MT']['std']:.6f} | "
          f"asym {g4['fro_asym_ratio']['mean']:.6e}"
          f" (group-mean-matrix asym {gm_v4['fro_asym_ratio']:.6e})", flush=True)
    print(f"[sym] n_subjects: {len(rows)} | elapsed {time.time() - t0:.1f}s "
          f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
