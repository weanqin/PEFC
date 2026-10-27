"""|EC_pert(unbiased μ=0, q=0.25)| × GC-EC 逐被试相关性（2026-10-21 用户指令）.

协议与 gen_ec1_abs_corr.py（10-21-v5main-ec-explore）一致：
  GC refs = refs/granger_bivariate_spectral/edrav4-46-resample250-.../refs_s{id}.npz
  （ec[a,b]=a→b，行=源/列=靶，对角 0；与 EC_pert 同向 → aligned = |conn| vs ec，flipped = vs ec.T）
  offdiag 3782 对，Spearman + Pearson，逐被试 + 组汇总 + 组均值矩阵。
"""

import glob
import json

import numpy as np
from scipy import stats as sps

OUT = "/root/npi/outputs/10-21-ec-globaltau-v5main/unbiased_mu0_ec"
REFS_GLOB = ("/root/npi/refs/granger_bivariate_spectral/"
             "edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/refs_s*.npz")
TEST_IDS = [12, 17, 18, 24, 30, 38, 44, 55, 58, 61]


def main():
    refs = {}
    for f in sorted(glob.glob(REFS_GLOB)):
        z = np.load(f, allow_pickle=True)
        refs[int(z["subject_id"])] = z["ec"].astype(np.float64)
    off = ~np.eye(62, dtype=bool)

    per = {}
    stack_abs, stack_ec = [], []
    for i, sid in enumerate(TEST_IDS):
        d = json.load(open(f"{OUT}/subject_{i}.json"))
        assert d["subject_file"] == f"s{sid}", f"subject_{i} 非预期 s{sid}"
        conn = np.array(d["connectivity_matrix"], dtype=np.float64)
        A = np.abs(conn)
        ec = refs[sid]
        r = {"spearman_aligned": float(sps.spearmanr(A[off], ec[off]).statistic),
             "pearson_aligned": float(np.corrcoef(A[off], ec[off])[0, 1]),
             "spearman_flipped": float(sps.spearmanr(A[off], ec.T[off]).statistic),
             "pearson_flipped": float(np.corrcoef(A[off], ec.T[off])[0, 1]),
             "tau": d["tau_global"], "pct_interior": d["diagnostics"]["pct_interior"]}
        per[sid] = r
        stack_abs.append(A)
        stack_ec.append(ec)
        print(f"s{sid}: |EC_pert| x GC rho={r['spearman_aligned']:+.4f} "
              f"r={r['pearson_aligned']:+.4f} (flip rho={r['spearman_flipped']:+.4f}, "
              f"r={r['pearson_flipped']:+.4f}) interior={r['pct_interior']:.1f}%", flush=True)

    def agg(k):
        a = np.array([per[s][k] for s in per])
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max())}

    A_mean = np.stack(stack_abs).mean(axis=0)
    ec_mean = np.stack(stack_ec).mean(axis=0)
    summary = {
        "naming": "EC_pert = unbiased mu=0 global_tau (q=0.25@sigma_ref=0.3) connectivity, "
                  "gpfm-mamba-v5-main surrogate",
        "subject": "|EC_pert| vs GC-EC, offdiag 3782",
        "reference": "edrav4-46-resample250 (任务 10-21-oldlib-gc-edrav4)",
        "alignment": "aligned = row=source vs ec; flipped = vs ec.T",
        "per_subject": {str(k): v for k, v in per.items()},
        "group": {k: agg(k) for k in
                  ("spearman_aligned", "pearson_aligned",
                   "spearman_flipped", "pearson_flipped")},
        "group_mean": {
            "spearman_aligned": float(sps.spearmanr(A_mean[off], ec_mean[off]).statistic),
            "pearson_aligned": float(np.corrcoef(A_mean[off], ec_mean[off])[0, 1])},
    }
    with open(f"{OUT}/refs_corr_10_summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    g = summary["group"]
    print(f"\nGROUP aligned:  rho {g['spearman_aligned']['mean']:+.4f}"
          f"±{g['spearman_aligned']['std']:.4f} "
          f"[{g['spearman_aligned']['min']:+.4f},{g['spearman_aligned']['max']:+.4f}] | "
          f"r {g['pearson_aligned']['mean']:+.4f}±{g['pearson_aligned']['std']:.4f}")
    print(f"GROUP flipped:  rho {g['spearman_flipped']['mean']:+.4f}"
          f"±{g['spearman_flipped']['std']:.4f} | "
          f"r {g['pearson_flipped']['mean']:+.4f}±{g['pearson_flipped']['std']:.4f}")
    print("group_mean:", json.dumps(summary["group_mean"]))


if __name__ == "__main__":
    main()
