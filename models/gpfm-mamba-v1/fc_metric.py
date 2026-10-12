"""指标 1（口径 v2，用户 2026-10-12 指定）：Pearson FC 矩阵一致性。

口径（evaluation-calibers 变更纪律：旧 GC 口径结果保留不动，见 experiment.md Run 007/008）：
- FC 矩阵：逐受试者在 future 128 点段（拼接该受试者全部窗）上计算 62×62 Pearson 相关
  → 组内受试者平均 → 每组一个 FC 矩阵（真实组 vs 生成组，两组完全相同流程）
- 一致性：上三角（含对角线剔除，1891 项）展平 → Pearson r + Spearman ρ（秩变换后 Pearson）
- 置信度：受试者级**配对** bootstrap（两组同重采样索引，1000 次，seed=2026）→ 95% CI
- 输入：eval/gc_input.npz（gen/real (N,62,128) + win_subj；生成侧=seed 1235 确定性 K=1 样本，
  与 GC 口径同一输入，保证与旧口径可比的「同输入、不同度量」）

运行：python3 models/gpfm-mamba-v1/fc_metric.py --run-id main-01
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main-01")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    base = Path("/root/npi/outputs/gpfm-mamba-f0") / args.run_id
    d = np.load(base / "eval" / "gc_input.npz")
    gen, real, subj = d["gen"], d["real"], d["win_subj"]
    subjects = np.unique(subj)
    assert len(subjects) >= 2, "受试者级 bootstrap 需 ≥2 例受试者"

    t0 = time.time()
    # 逐受试者 FC（两组同流程）
    fc_real_s = np.stack([subject_fc(real[subj == s]) for s in subjects])   # (S,62,62)
    fc_gen_s = np.stack([subject_fc(gen[subj == s]) for s in subjects])
    FC_real = fc_real_s.mean(axis=0)
    FC_gen = fc_gen_s.mean(axis=0)

    a, b = tri_flat(FC_gen), tri_flat(FC_real)
    r = pearson_r(a, b)
    rho = spearman_rho(a, b)

    # 受试者级配对 bootstrap：同索引重采样两组的受试者级 FC，重算组平均后的一致性
    rng = np.random.default_rng(args.seed)
    S = len(subjects)
    rs, rhos = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, S, S)
        fa = tri_flat(fc_gen_s[idx].mean(axis=0))
        fb = tri_flat(fc_real_s[idx].mean(axis=0))
        rs.append(pearson_r(fa, fb))
        rhos.append(spearman_rho(fa, fb))
    r_ci = (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))
    rho_ci = (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))

    out = {
        "metric": "指标 1 v2：Pearson FC 矩阵一致性（逐受试者 FC → 组平均 → 上三角 r/ρ）",
        "run_id": args.run_id,
        "n_subjects": int(S),
        "n_windows": int(len(subj)),
        "n_edges_upper": int(len(a)),
        "pearson_r": round(r, 5),
        "spearman_rho": round(rho, 5),
        "pearson_ci95": [round(x, 5) for x in r_ci],
        "spearman_ci95": [round(x, 5) for x in rho_ci],
        "bootstrap": {"unit": "subject-paired", "n_boot": args.n_boot, "seed": args.seed},
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
