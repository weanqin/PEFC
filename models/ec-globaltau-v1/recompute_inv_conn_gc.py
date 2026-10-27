"""conn 置换为 1/(σ*+1e-4) 的 EC 重算 + GC-EC 相关性（2026-10-21 用户指令）.

输入：unbiased_mu0_ec/subject_{0..9}.json 的 σ*（无偏 μ=0，q=0.25@0.3，已 GPU 计算）。
新公式：conn = 1/(σ* + 1e-4)（单调递减于 σ*，同向于旧式 −(log σ*+ln2πe)+1；σ*→0 时
conn→10^4，σ*=0.48 时 conn≈2.08）。sign/元数据随档拷贝；纯离线变换，无 GPU。

注意：Spearman 对严格单调变换不变 → conn_new vs GC 的 Spearman 应与 raw conn vs GC 逐位
一致（脚本内置断言校验）；Pearson 会改变。相关性协议同 gen_unbiased_gc_corr.py：
abs 与 raw 两口径 × aligned/flipped × Spearman/Pearson。

用法：
    /root/miniconda3/bin/python models/ec-globaltau-v1/recompute_inv_conn_gc.py
"""

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

SRC = Path("/root/npi/outputs/10-21-ec-globaltau-v5main/unbiased_mu0_ec")
OUT = Path("/root/npi/outputs/10-21-ec-globaltau-v5main/unbiased_mu0_ec_inv")
REFS_GLOB = ("/root/npi/refs/granger_bivariate_spectral/"
             "edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/refs_s*.npz")
TEST_IDS = [12, 17, 18, 24, 30, 38, 44, 55, 58, 61]
EPS = 1e-4


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {}
    for f in sorted(glob.glob(REFS_GLOB)):
        z = np.load(f, allow_pickle=True)
        refs[int(z["subject_id"])] = z["ec"].astype(np.float64)
    off = ~np.eye(62, dtype=bool)

    per = {}
    stack_abs, stack_raw, stack_ec = [], [], []
    for i, sid in enumerate(TEST_IDS):
        d = json.load(open(SRC / f"subject_{i}.json"))
        assert d["subject_file"] == f"s{sid}"
        sigma = np.array(d["sigma_matrix"], dtype=np.float64)
        conn = 1.0 / (sigma + EPS)
        sign = np.array(d["sign_matrix"], dtype=np.float64)

        payload = dict(d)  # 元数据随档拷贝（hparams/surrogate/diagnostics/τ 等原样）
        payload.update({
            "connectivity_matrix": conn.tolist(),
            "connectivity_formula": "1/(sigma*+1e-4)",
            "signed_connectivity_matrix": (sign * conn).tolist(),
            "method": d["method"] + " | conn remapped to 1/(sigma*+1e-4)",
            "recompute_note": ("offline monotone remap of unbiased sigma* "
                               "(no new forwards); Spearman invariant check built in"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        with open(OUT / f"subject_{i}.json", "w") as fh:
            json.dump(payload, fh, indent=2)

        A, R = np.abs(conn), conn
        ec = refs[sid]
        r = {"spearman_abs_aligned": float(sps.spearmanr(A[off], ec[off]).statistic),
             "pearson_abs_aligned": float(np.corrcoef(A[off], ec[off])[0, 1]),
             "spearman_raw_aligned": float(sps.spearmanr(R[off], ec[off]).statistic),
             "pearson_raw_aligned": float(np.corrcoef(R[off], ec[off])[0, 1]),
             "spearman_abs_flipped": float(sps.spearmanr(A[off], ec.T[off]).statistic),
             "spearman_raw_flipped": float(sps.spearmanr(R[off], ec.T[off]).statistic)}
        per[sid] = r
        stack_abs.append(A); stack_raw.append(R); stack_ec.append(ec)
        print(f"s{sid}: raw rho={r['spearman_raw_aligned']:+.4f} "
              f"r={r['pearson_raw_aligned']:+.4f} | abs rho={r['spearman_abs_aligned']:+.4f} "
              f"r={r['pearson_abs_aligned']:+.4f} (flip raw rho="
              f"{r['spearman_raw_flipped']:+.4f})", flush=True)

    # 单调不变性校验：新旧公式 raw Spearman 应逐位一致（conn_new 严格单调于 conn_old）
    log2pie = np.log(2.0 * np.pi * np.e)
    for i, sid in enumerate(TEST_IDS):
        d = json.load(open(SRC / f"subject_{i}.json"))
        conn_old = -(np.log(np.array(d["sigma_matrix"], dtype=np.float64)) + log2pie) + 1.0
        r_old = sps.spearmanr(conn_old[off], refs[sid][off]).statistic
        assert abs(r_old - per[sid]["spearman_raw_aligned"]) < 1e-9, \
            f"s{sid} 单调不变性校验失败: {r_old} vs {per[sid]['spearman_raw_aligned']}"
    print("[check] Spearman 单调不变性（vs 旧式 −log σ*）: 10/10 PASS", flush=True)
    def agg(k):
        a = np.array([per[s][k] for s in per])
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max())}

    A_mean = np.stack(stack_abs).mean(axis=0)
    R_mean = np.stack(stack_raw).mean(axis=0)
    ec_mean = np.stack(stack_ec).mean(axis=0)
    summary = {
        "naming": "EC_pert = 1/(sigma*+1e-4) on unbiased mu=0 sigma* (q=0.25@sigma_ref=0.3)",
        "subject": "conn vs GC-EC, offdiag 3782; raw (no abs) and abs variants",
        "reference": "edrav4-46-resample250 (任务 10-21-oldlib-gc-edrav4)",
        "alignment": "aligned = row=source vs ec; flipped = vs ec.T",
        "monotonicity_note": ("conn=1/(sigma*+eps) is strictly decreasing in sigma*, same "
                              "direction as -(log sigma*+ln2pie)+1 → Spearman(raw) identical "
                              "to old-formula raw Spearman; only Pearson/abs change"),
        "per_subject": {str(k): v for k, v in per.items()},
        "group": {k: agg(k) for k in next(iter(per.values()))},
        "group_mean": {
            "spearman_raw_aligned": float(sps.spearmanr(R_mean[off], ec_mean[off]).statistic),
            "pearson_raw_aligned": float(np.corrcoef(R_mean[off], ec_mean[off])[0, 1]),
            "spearman_abs_aligned": float(sps.spearmanr(A_mean[off], ec_mean[off]).statistic),
            "pearson_abs_aligned": float(np.corrcoef(A_mean[off], ec_mean[off])[0, 1])},
    }
    with open(OUT / "refs_corr_10_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    g = summary["group"]
    print(f"\nGROUP raw:  rho {g['spearman_raw_aligned']['mean']:+.4f}"
          f"±{g['spearman_raw_aligned']['std']:.4f} | "
          f"r {g['pearson_raw_aligned']['mean']:+.4f}±{g['pearson_raw_aligned']['std']:.4f}")
    print(f"GROUP abs:  rho {g['spearman_abs_aligned']['mean']:+.4f}"
          f"±{g['spearman_abs_aligned']['std']:.4f} | "
          f"r {g['pearson_abs_aligned']['mean']:+.4f}±{g['pearson_abs_aligned']['std']:.4f}")
    print("group_mean:", json.dumps(summary["group_mean"]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
