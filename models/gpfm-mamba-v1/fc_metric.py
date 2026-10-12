"""指标 1（口径 v2.1，用户 2026-10-12 指定）：Pearson FC 矩阵一致性。

口径（evaluation-calibers 变更纪律：旧口径结果保留不动，见 experiment.md Run 007/008/009）：
- FC 矩阵：逐受试者在 future 128 点段（拼接该受试者全部窗）上计算 62×62 Pearson 相关
  → 组内受试者平均 → 每组一个 FC 矩阵（真实组 vs 生成组，两组完全相同流程）
- 一致性点估计：上三角（对角剔除，1891 项）展平 → Pearson r + Spearman ρ（组平均矩阵）
- **显著性（v2.1，替代 v2 的受试者级配对 bootstrap CI）：逐受试者一致性值 r_s/ρ_s
  （同受试者 gen FC vs real FC 的相关）→ 单样本双侧 t 检验（H0: 均值=0，df=S−1）**
- 输入：eval/gc_input.npz（gen/real (N,62,128) + win_subj；生成侧=seed 1235 确定性 K=1 样本，
  与 GC 口径同一输入，保证与旧口径可比的「同输入、不同度量」）

运行：python3 models/gpfm-mamba-v1/fc_metric.py --run-id main-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gc_psd import t_two_sided_p  # noqa: E402  同变体内复用（Lentz 连分式 incomplete beta）


def subject_fc(x_subj: np.ndarray) -> np.ndarray:
    """单个受试者全部 future 段拼接后的 62×62 Pearson 相关（float64）。"""
    x = x_subj.transpose(1, 0, 2).reshape(x_subj.shape[1], -1).astype(np.float64)  # (C, N*T)
    return np.corrcoef(x)


def tri_flat(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson_r(ra, rb)


def one_sample_ttest(x: np.ndarray):
    """单样本双侧 t 检验（H0: 均值=0）；返回 (t, df, p, mean, sd)。"""
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / np.sqrt(n))
    return t, n - 1, t_two_sided_p(t, n - 1), m, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main-01")
    args = ap.parse_args()

    base = Path("/root/npi/outputs/gpfm-mamba-f0") / args.run_id
    d = np.load(base / "eval" / "gc_input.npz")
    gen, real, subj = d["gen"], d["real"], d["win_subj"]
    subjects = np.unique(subj)
    assert len(subjects) >= 2, "逐受试者 t 检验需 ≥2 例受试者"

    t0 = time.time()
    # 逐受试者 FC（两组同流程）
    fc_real_s = np.stack([subject_fc(real[subj == s]) for s in subjects])   # (S,62,62)
    fc_gen_s = np.stack([subject_fc(gen[subj == s]) for s in subjects])
    FC_real = fc_real_s.mean(axis=0)
    FC_gen = fc_gen_s.mean(axis=0)

    # 点估计：组平均矩阵的上三角一致性
    a, b = tri_flat(FC_gen), tri_flat(FC_real)
    r = pearson_r(a, b)
    rho = spearman_rho(a, b)

    # 显著性（v2.1）：逐受试者一致性值的单样本双侧 t 检验（df = S-1）
    r_s = np.array([pearson_r(tri_flat(fc_gen_s[i]), tri_flat(fc_real_s[i])) for i in range(len(subjects))])
    rho_s = np.array([spearman_rho(tri_flat(fc_gen_s[i]), tri_flat(fc_real_s[i])) for i in range(len(subjects))])
    t_r, df_r, p_r, r_mean, r_sd = one_sample_ttest(r_s)
    t_rho, df_rho, p_rho, rho_mean, rho_sd = one_sample_ttest(rho_s)

    out = {
        "metric": "指标 1 v2.1：Pearson FC 矩阵一致性（点估计=组平均矩阵；显著性=逐受试者单样本 t 检验）",
        "run_id": args.run_id,
        "n_subjects": int(len(subjects)),
        "n_windows": int(len(subj)),
        "n_edges_upper": int(len(a)),
        "pearson_r": round(r, 5),
        "spearman_rho": round(rho, 5),
        "significance": {
            "method": "one-sample two-sided t-test on per-subject consistency (H0: mean=0)",
            "pearson_r_per_subject_mean": round(r_mean, 5),
            "pearson_r_per_subject_sd": round(r_sd, 5),
            "pearson_t": round(t_r, 4),
            "pearson_df": int(df_r),
            "pearson_p": p_r,
            "spearman_rho_per_subject_mean": round(rho_mean, 5),
            "spearman_rho_per_subject_sd": round(rho_sd, 5),
            "spearman_t": round(t_rho, 4),
            "spearman_df": int(df_rho),
            "spearman_p": p_rho,
            "pearson_r_per_subject": [round(x, 5) for x in r_s.tolist()],
        },
        "elapsed_s": round(time.time() - t0, 2),
        "input": "eval/gc_input.npz（生成侧 = gc_seed 1235 确定性 K=1 样本，与 GC 口径同输入）",
    }
    out_path = base / "eval" / "fc_metric.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savez(base / "eval" / "fc_matrices.npz", FC_real=FC_real, FC_gen=FC_gen)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"落盘 {out_path}")


if __name__ == "__main__":
    main()
