#!/usr/bin/env python3
"""recompute_conn_v4.py — conn v4 重映射 + GC 对照（任务 10-24-exitgnn-ec-unbiased Run 003）.

用户拍板交付公式（2026-10-24）：conn_v4 = −(ln σ* + 0.5·ln 2πe) + 5/7。

数学事实（先例 w500 Run 007 在档，本脚本断言组封死口径漂移）：
  v4 = 房内 v1（−(ln σ*+ln2πe)+1）+ 常数 Δc，Δc = 0.5·ln2πe − 2/7 ≈ +1.1332
  ⇒ raw 口径 Spearman/Pearson 对任意参考不变（同 σ* float64 路径断言 |Δ|<1e-9）；
  json conn = 主跑 float32 舍入值（默认公式 2026-10-27 切 v4 前为 v1、之后为 v4，
  按档 connectivity_formula 声明自适应取对照式），与 float64 重算存在 ~1e-13 量级
  Spearman 尾差 [舍入引入新并列，s5 实测]——精度伪影非口径漂移，舍入差随档登记；
  零点 σ* = e^(5/7−0.5·ln2πe) = 0.4943 落在网格上界 0.48 之外 ⇒ conn 全正
  （min = conn(0.48) = +0.0293 > 0）⇒ **abs ≡ raw**（逐位断言）。

GC 参考（2026-10-27 线配置固定：v2 1-125hz 对比项经用户拍板取消，仅存 v1 单参考）：
  v1_1_45_fixed8  = refs mild40-trial500-zscore-fs250-fixed8-f1-45-favg（1–45 Hz 固定阶 8）
协议：offdiag 16256 对；aligned（行=源 vs refs ec）/ flipped（vs ec.T）×
Spearman/Pearson；描述性登记不构成代理脑资格判定；与 w500 Run 007 先例并置为
**跨代理脑描述性并置**（θ 不同，差异 = 代理脑差异 + 噪声，非复现性判读）。

输入：ec_main/subject_*.json（Run 001，σ* + conn[2026-10-27 起新档为 v4、旧档
v1] + sign/元数据）。
输出：recompute_v4/subject_*.json（重映射档）+ gccorr_dual_summary.json（参考块）。
纯离线，无 GPU；元数据随档拷贝（先例 recompute_inv_conn_gc.py 同款）。

Run:
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/recompute_conn_v4.py
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

DEFAULT_MAIN = "/root/npi/outputs/10-24-exitgnn-ec-unbiased/ec_main"
DEFAULT_OUT = "/root/npi/outputs/10-24-exitgnn-ec-unbiased/recompute_v4"
# 2026-10-27 线配置固定：v2_1_125_aic4_25（mild40 aic4-25-f1-125 档）对比项经
# 用户拍板取消（refs 物理文件已删）；仅保留 v1 单参考
REFS = {
    "v1_1_45_fixed8":
        "/root/npi/refs/granger_bivariate_spectral/"
        "mild40-trial500-zscore-fs250-fixed8-f1-45-favg",
}
HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)
V4_OFFSET = 5.0 / 7.0
DEFAULT_TASK_LABEL = ("10-24-exitgnn-ec-unbiased Run 003 "
                      "(v4 remap + descriptive EC x GC)")
DEFAULT_SURROGATE_NAME = "exit-gnn variant surrogate"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--main-dir", type=str, default=DEFAULT_MAIN)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--task-label", type=str, default=DEFAULT_TASK_LABEL,
                   help="输出 gccorr_dual_summary.json 的 task 字段（默认 = 先例"
                        "常量串，旧调用面不变；新任务传本任务标签）")
    p.add_argument("--surrogate-name", type=str, default=DEFAULT_SURROGATE_NAME,
                   help="naming 串中代理脑描述段（默认 = 先例常量串，旧调用面"
                        "不变；新任务传本任务代理脑描述，元数据专用零计算改动）")
    return p.parse_args(argv)


def load_refs(dirpath: str) -> dict:
    refs = {}
    for f in sorted(Path(dirpath).glob("refs_s*.npz")):
        z = np.load(f, allow_pickle=True)
        refs[int(z["subject_id"])] = z["ec"].astype(np.float64)
    assert refs, f"no refs_s*.npz under {dirpath}"
    return refs


def main() -> int:
    a = parse_args()
    main_dir, out_dir = Path(a.main_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    off = ~np.eye(128, dtype=bool)
    refs_all = {name: load_refs(p) for name, p in REFS.items()}
    rows = sorted(main_dir.glob("subject_*.json"))
    assert rows, f"no subject_*.json under {main_dir}"

    per_refs = {name: {} for name in REFS}
    stack_conn = {name: [] for name in REFS}
    stack_ec = {name: [] for name in REFS}
    n_checks = 0
    json_rounding_max = 0.0

    t0 = time.time()
    for f in rows:
        d = json.load(open(f))
        sid = int(d["subject_file"][1:])
        sigma = np.array(d["sigma_matrix"], dtype=np.float64)
        conn_json = np.array(d["connectivity_matrix"], dtype=np.float64)
        sign = np.array(d["sign_matrix"], dtype=np.float64)

        # ---- v4 重映射 + 断言组（8/8 全过才进汇总）----
        # 不变性断言须在同一条 float64 σ* 路径上做：json conn 是主跑 float32 舍入值
        # （默认公式 2026-10-27 切 v4 前为 v1、之后为 v4，按档公式声明自适应对照），
        # 与 float64 重算相比存在 ~1e-13 量级 Spearman 尾差（舍入引入新并列，s5 实测
        # 1−ρ=7.3e-13）——精度伪影非口径漂移；json 舍入差另作诊断登记。
        conn_v1_re = -(np.log(sigma) + 2.0 * HALF_LOG_2PIE) + 1.0  # 房内 v1 同 σ* 重算
        conn_v4 = -(np.log(sigma) + HALF_LOG_2PIE) + V4_OFFSET
        assert abs(sps.spearmanr(conn_v4[off], conn_v1_re[off]).statistic - 1.0) < 1e-9, \
            f"{d['subject_file']}: v4 非严格单调于 v1（同 σ* float64 路径）"
        assert conn_v4.min() > 0.0, \
            f"{d['subject_file']}: conn_v4 非全正 min={conn_v4.min()}"
        assert np.array_equal(np.abs(conn_v4), conn_v4), \
            f"{d['subject_file']}: abs != raw（全正前提下必逐位相等）"
        # json conn 公式版本自适应：新档 v4 / 旧档 v1，按 connectivity_formula 声明
        # 取对照式（错配会表现为 ~1.13 平移量级而非舍入量级，可即时识别）
        json_is_v4 = "5/7" in str(d.get("connectivity_formula", ""))
        json_ref = conn_v4 if json_is_v4 else conn_v1_re
        json_rounding = float(np.max(np.abs(conn_json - json_ref)))
        json_rounding_max = max(json_rounding_max, json_rounding)
        n_checks += 3

        payload = dict(d)  # 元数据随档拷贝（hparams/surrogate/τ/diagnostics 原样）
        payload.update({
            "connectivity_matrix": conn_v4.tolist(),
            "connectivity_formula": "-(log sigma* + 0.5*log(2*pi*e)) + 5/7",
            "signed_connectivity_matrix": (sign * conn_v4).tolist(),
            "method": d["method"] + " | conn remapped to v4 = -(ln sigma*+0.5*ln2pie)+5/7",
            "recompute_note": (
                "offline strict constant-shift remap (d = +0.5*ln2pie - 2/7); raw "
                "Spearman/Pearson bitwise-invariant vs house v1 formula; zero point "
                "sigma*=0.4943 outside grid [0.01,0.48] => strictly positive => "
                "abs == raw (bitwise asserted)"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        with open(out_dir / f.name, "w") as fh:
            json.dump(payload, fh, indent=2)

        # ---- GC 对照（协议对齐 gen_unbiased_gc_corr.py / Run 007 先例；单参考 v1）----
        for name, refs in refs_all.items():
            assert sid in refs, f"refs {name} 缺 s{sid}"
            ec = refs[sid]
            # 常数平移 Pearson 同精度不变性（Spearman 已由断言 1 覆盖；同 σ* float64 路径）
            r_v4 = float(np.corrcoef(conn_v4[off], ec[off])[0, 1])
            r_v1 = float(np.corrcoef(conn_v1_re[off], ec[off])[0, 1])
            assert abs(r_v4 - r_v1) < 1e-9, \
                f"{d['subject_file']}/{name}: Pearson 平移不变性 {r_v4} vs {r_v1}"
            n_checks += 1
            A, R = np.abs(conn_v4), conn_v4  # A ≡ R（断言 3）；两键都算 = 协议形状保留
            r = {
                "subject": d["subject_file"], "group": d.get("group"),
                "spearman_abs_aligned": float(sps.spearmanr(A[off], ec[off]).statistic),
                "pearson_abs_aligned": float(np.corrcoef(A[off], ec[off])[0, 1]),
                "spearman_raw_aligned": float(sps.spearmanr(R[off], ec[off]).statistic),
                "pearson_raw_aligned": float(np.corrcoef(R[off], ec[off])[0, 1]),
                "spearman_abs_flipped": float(sps.spearmanr(A[off], ec.T[off]).statistic),
                "spearman_raw_flipped": float(sps.spearmanr(R[off], ec.T[off]).statistic),
            }
            per_refs[name][sid] = r
            stack_conn[name].append(conn_v4)
            stack_ec[name].append(ec)
            print(f"[{name}] s{sid} ({d.get('group')}): "
                  f"rho={r['spearman_raw_aligned']:+.4f} "
                  f"r={r['pearson_raw_aligned']:+.4f} "
                  f"(flip rho={r['spearman_raw_flipped']:+.4f})", flush=True)

    def agg(per, k):
        v = np.array([per[s][k] for s in per], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    blocks = {}
    for name in REFS:
        per = per_refs[name]
        conn_mean = np.stack(stack_conn[name]).mean(axis=0)
        ec_mean = np.stack(stack_ec[name]).mean(axis=0)
        blocks[name] = {
            "refs_dir": REFS[name],
            "refs_caliber": ("mne-connectivity spectral gc, VAR order 8 fixed, "
                             "1-45 Hz faverage"),
            "per_subject": {str(k): v for k, v in per.items()},
            "group": {k: agg(per, k) for k in next(iter(per.values()))
                      if k.startswith(("spearman", "pearson"))},
            "group_mean": {
                "spearman_raw_aligned":
                    float(sps.spearmanr(conn_mean[off], ec_mean[off]).statistic),
                "pearson_raw_aligned":
                    float(np.corrcoef(conn_mean[off], ec_mean[off])[0, 1])},
        }

    summary = {
        "task": a.task_label,
        "naming": "EC_pert = v4 formula -(ln sigma*+0.5*ln2pie)+5/7 on unbiased mu=0 "
                  "global_tau sigma* (q=0.5@sigma_ref=0.3), latent-flow mild "
                  f"{a.surrogate_name} (provisional, see surrogate_caveat in "
                  "subject jsons); abs == raw (all positive, bitwise asserted)",
        "formula_facts": {
            "constant_shift_vs_v1": float(HALF_LOG_2PIE - 2.0 / 7.0),
            "zero_point_sigma": float(math.exp(V4_OFFSET - HALF_LOG_2PIE)),
            "invariance": ("raw Spearman/Pearson invariant vs v1 on the same float64 "
                           "sigma* path (asserted per subject per refs set, tol 1e-9); "
                           "abs==raw asserted bitwise"),
            "json_f32_rounding_maxabs": float(json_rounding_max),
        },
        "protocol": ("offdiag 16256 pairs; aligned = row=source vs ec; flipped = vs "
                     "ec.T; Spearman/Pearson; abs and raw blocks identical by "
                     "construction (kept for protocol shape)"),
        "descriptive_only": ("描述性并置登记（先例协议），不构成代理脑资格判定；"
                             "EC=代理脑内部量 vs GC=观测数据量，实现口径隔离"),
        "juxtaposition_note": (
            "v1 块与 w500 Run 007 先例（+0.1674±0.1661）的并置为跨代理脑描述性并置"
            "（θ 不同：w500 基座 vs exit-gnn 变体），数字差异 = 代理脑差异 + 噪声，"
            "不作复现性判读"),
        "asserts_passed": int(n_checks),
        "refs_blocks": blocks,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "gccorr_dual_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    for name in REFS:
        g = blocks[name]["group"]
        print(f"\n[{name}] GROUP raw aligned: "
              f"rho {g['spearman_raw_aligned']['mean']:+.4f}"
              f"±{g['spearman_raw_aligned']['std']:.4f} | "
              f"r {g['pearson_raw_aligned']['mean']:+.4f}"
              f"±{g['pearson_raw_aligned']['std']:.4f} | "
              f"group_mean rho={blocks[name]['group_mean']['spearman_raw_aligned']:+.4f}",
              flush=True)
    print(f"\n[v4] asserts passed: {n_checks} | n_subjects: {len(rows)} | "
          f"elapsed {time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
