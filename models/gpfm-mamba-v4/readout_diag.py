#!/usr/bin/env python3
"""readout_diag.py — v4-B cMLP head 内部有向 readout 矩阵诊断（R6，仅诊断、不参与主判据）。

定义（design §6，全仓库方向约定：行=源、列=靶）：
  M[j, i] = ‖W^{(i),1}_{:,j}‖_F
  W1 为 checkpoint（best，EMA 权重）head 第一层堆叠张量，形状 (62 目标 i, d_z, 62·d_ch)，
  末维按源通道 j 列分块 [j·d_ch:(j+1)·d_ch)（model.py [4] 注记）——对 (d_z, d_ch) 块求
  Frobenius 范数得 N[i,j]，转置成 M[j,i]（行=源 j、列=靶 i，与 EC/GC 矩阵同向）。
  d_ch=1 拍板口径下 N[i,j] 即 torch.linalg.norm(W1, dim=1)——脚本内双路径 allclose 断言。

比较（仅诊断）：M 与 evaluate 落盘的外部 GC 矩阵（eval/gc_gen.npy 生成侧 / gc_real.npy
真实侧）的展平 Spearman（+ Pearson 参考）：
  - 主口径 = 去对角 62×61=3782 有向边（与 GC 一致性口径同构；GC 对角恒 0 无自指语义）；
  - 辅口径 = 全 62×62 展平（含 M 对角自项）——两口径数值均落盘登记。
无任何监督项参与训练（损失与 v1/v2 逐字一致）；group lasso 为条件后续（prd R6）。

用法：python3 models/gpfm-mamba-v4/readout_diag.py --run-id v4b-main-01
依赖：torch（读 ckpt）、numpy、common/visualizer（ec_matrix 图类型）、v1 gc_psd 统计函数
      （pearson_r/spearman_rho，只读复用）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# BLAS 线程上限（environment.md 惯例，numpy 导入前）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_V1 = _HERE.parent / "gpfm-mamba-v1"
sys.path.insert(0, str(_V1))
sys.path.insert(0, str(_REPO))

from gc_psd import offdiag_flat, pearson_r, spearman_rho  # noqa: E402  v1 只读复用

RUN_PREFIX = "v4b-"
OUTPUTS_ROOT = Path("/root/npi/outputs/gpfm-mamba-f0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=str, required=True, help=f"run 目录名（{RUN_PREFIX} 前缀）")
    p.add_argument("--ckpt", type=str, default=None, help="显式 ckpt（默认 best）")
    a = p.parse_args()
    if not a.run_id.startswith(RUN_PREFIX):
        sys.exit(f"[readout] --run-id 须以 '{RUN_PREFIX}' 开头，got: {a.run_id}")
    return a


def main() -> None:
    args = parse_args()
    run_dir = OUTPUTS_ROOT / args.run_id
    ckpt_path = args.ckpt or str(run_dir / "checkpoints" / "gpfm-mamba-v4b-best.pt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[ckpt] {ckpt_path}")
    print(f"[ckpt] variant={ck.get('variant')} epoch={ck.get('epoch')} "
          f"best_val={ck.get('best_val'):.5f} config_hash={ck.get('config_hash')}")
    sd = ck["ema"]["shadow"]                      # 评估用 EMA 权重（best 判据口径）
    print(f"[ckpt] EMA shadow keys/shapes:")
    for k in ("W1", "b1", "W2", "b2", "head_norm.weight"):
        print(f"        {k}: {tuple(sd[k].shape)}")

    W1 = sd["W1"].float()                         # (62 i, d_z, 62·d_ch)
    n_ch = W1.shape[0]
    d_z = W1.shape[1]
    k_head = W1.shape[2]
    assert k_head % n_ch == 0, f"W1 末维 {k_head} 非 {n_ch} 整数倍"
    d_ch = k_head // n_ch
    print(f"[shape] W1 (62 i, d_z={d_z}, 62·d_ch={k_head}) -> d_ch={d_ch}")

    # 主路径：按 (i, d_z, j, d_ch) 分块范数 -> N[i,j] -> M[j,i]（行=源、列=靶）
    blocks = W1.reshape(n_ch, d_z, n_ch, d_ch)                      # (i, d_z, j, d_ch)
    N = torch.linalg.norm(blocks, dim=(1, 3)).numpy()               # (i, j) Frobenius over (d_z,d_ch)
    M = N.T                                                         # M[j, i]
    # d_ch=1 时与直接对 d_z 求范数逐位一致（双路径断言）
    if d_ch == 1:
        N_direct = torch.linalg.norm(W1, dim=1).numpy()
        assert np.allclose(N, N_direct), "d_ch=1 双路径范数不一致"
        print("[shape] d_ch=1 双路径（分块范数 vs 直接 norm(dim=1)）allclose 通过")
    print(f"[M] shape {M.shape}（行=源 j、列=靶 i）| min {M.min():.5f} max {M.max():.5f} "
          f"mean {M.mean():.5f} | 对角（自源）均值 {np.diag(M).mean():.5f} vs 非对角均值 "
          f"{offdiag_flat(M).mean():.5f}")

    # ---- 与外部 GC 矩阵比较（evaluate 落盘产物；仅诊断）----
    gc_gen = np.load(run_dir / "eval" / "gc_gen.npy")
    gc_real = np.load(run_dir / "eval" / "gc_real.npy")
    res = {"run_id": args.run_id, "ckpt": ckpt_path, "epoch": int(ck["epoch"]),
           "config_hash": ck["config_hash"], "weights": "EMA shadow（best 判据口径）",
           "M_def": "M[j,i] = ||W1[i,:,j·d_ch:(j+1)·d_ch]||_F（行=源 j、列=靶 i）；d_ch=%d、d_z=%d"
                    % (d_ch, d_z),
           "M_stats": {"min": round(float(M.min()), 5), "max": round(float(M.max()), 5),
                       "mean": round(float(M.mean()), 5),
                       "diag_mean": round(float(np.diag(M).mean()), 5),
                       "offdiag_mean": round(float(offdiag_flat(M).mean()), 5)}}
    for tag, G in (("gc_gen", gc_gen), ("gc_real", gc_real)):
        m_off, g_off = offdiag_flat(M), offdiag_flat(G)
        res[tag] = {
            "spearman_offdiag_3782": round(spearman_rho(m_off, g_off), 5),
            "pearson_offdiag_3782": round(pearson_r(m_off, g_off), 5),
            "spearman_full_62x62": round(spearman_rho(M.ravel(), G.ravel()), 5),
            "pearson_full_62x62": round(pearson_r(M.ravel(), G.ravel()), 5),
        }
        print(f"[cmp] M vs {tag}: Spearman(offdiag 3782) "
              f"{res[tag]['spearman_offdiag_3782']:.5f} | Pearson {res[tag]['pearson_offdiag_3782']:.5f}"
              f" | full 62×62 Spearman {res[tag]['spearman_full_62x62']:.5f}")
    res["caliber_note"] = ("主口径=去对角 3782 有向边（与 GC 一致性口径同构）；"
                           "全 62×62 辅口径含 M 对角自项（GC 对角恒 0）。仅诊断，不参与主判据")

    np.save(run_dir / "eval" / "readout_M.npy", M)
    out_json = run_dir / "eval" / "readout_diag.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # ---- 图（统一 visualizer：ec_matrix 三面板，行=源/列=靶）----
    from common.visualizer import Visualizer
    viz = Visualizer(str(run_dir / "figs"))
    fig = viz.render("ec_matrix", {
        "ec_a": M, "ec_b": gc_gen, "ec_c": gc_real,
        "label_a": "readout  M[j,i]=‖W¹_{:,j}‖_F", "label_b": "GC on generated (ln SSR ratio)",
        "label_c": "GC on real (ln SSR ratio)",
        "stats": {"M~GCgen rho": res["gc_gen"]["spearman_offdiag_3782"],
                  "M~GCreal rho": res["gc_real"]["spearman_offdiag_3782"]},
    }, f"eval_readout_matrix_{args.run_id}.png")
    print(f"[readout] 数值 {out_json} + M 矩阵 eval/readout_M.npy | 图 {fig}")


if __name__ == "__main__":
    main()
