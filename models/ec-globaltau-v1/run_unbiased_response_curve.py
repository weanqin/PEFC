"""无偏噪声 σ-响应曲线 — μ=0, 扰动 = σ·ξ, ξ~N(0,1)（用户拍板 2026-10-21）.

与 07-04 正典的差异（登记偏离）：GpfmProbeSite(use_biased_noise=False) —— 关闭固定
μ_bias=0.5·std 方向偏置，扰动恰为 σ·ξ（ξ 标准正态、seed+source 播种、同一 ξ 跨 σ 复用），
逐点加在 z-score 之后、入模之前的 x^p 通道 a 上。扰动注入/数据构造/seeds/EMA swap/bf16
全部复用既有实现，零源码改动。

执行：fold0 test 1 例（默认 s12）× 32 窗，逐源扫 log[0.01,0.48]×24 σ 网格 + σ_ref=0.3，
落盘 MSE 曲线并绘制 σ-响应（MSE）曲线；附带快速钳制诊断（τ=q 分位@0.3 时的 interior
比例，σ* 经 import 的 walk_monotone_curve）。

用法：
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      /root/miniconda3/bin/python models/ec-globaltau-v1/run_unbiased_response_curve.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/npi/models/latent-flow-v1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from latent_flow.attribution.bisect_sigma import (
    compute_global_tau,
    log_grid,
    walk_monotone_curve,
)
from latent_flow.scripts._connectivity_driver import subsample_trials

from adapter import FOLD0_TEST_IDS, load_surrogate, subject_global_windows
from probe_site import GpfmProbeSite

OUT_DIR = Path("/root/npi/outputs/10-21-ec-globaltau-v5main/unbiased_mu0_response")
SIGMA_MIN, SIGMA_MAX, N_GRID = 0.01, 0.48, 24       # 与阈值搜索同网格，便于直比
SIGMA_REF = 0.3

C_SERIES1 = "#2a78d6"
C_INK, C_INK2, C_GRID, C_SURFACE, C_CLAMP = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb", "#e34948"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = log_grid(SIGMA_MIN, SIGMA_MAX, N_GRID)
    cfg, model, ema, sampler, test_ds = load_surrogate(device=str(device))
    sid = FOLD0_TEST_IDS[args.subject_idx]
    widx = subject_global_windows(test_ds, sid)
    x_p_full = torch.stack([test_ds[gi][0] for gi in widx]).to(device)
    x_p, n_used = subsample_trials(x_p_full, args.n_trials, args.seed, args.subject_idx)
    N, C, P = x_p.shape
    F = sampler.prior.future_len
    print(f"s{sid}: windows {n_used} shape {tuple(x_p.shape)}", flush=True)

    gen = torch.Generator(device=device).manual_seed(args.seed * 10000 + sid)
    eps = torch.randn(N, 1, C, F, device=device, generator=gen)
    eta = torch.randn(N, 1, C, F, device=device, generator=gen)

    site = GpfmProbeSite(sampler, x_p, eps, eta, seed=args.seed, use_biased_noise=False)
    assert site._mu_bias is None, "μ_bias 未关闭"

    t0 = time.time()
    mse_curve = np.full((C, N_GRID, C), np.nan, dtype=np.float32)
    md_curve = np.full((C, N_GRID, C), np.nan, dtype=np.float32)
    resp_ref = np.full((C, C), np.nan, dtype=np.float32)
    with ema.swap(model):
        site.compute_baseline()
        for src in range(C):
            site.set_source(src)
            for gi, sig in enumerate(grid):
                mse_vec, md_vec = site.perturb_and_measure(float(sig), src)
                mse_curve[src, gi, :] = mse_vec.detach().cpu().numpy()
                md_curve[src, gi, :] = md_vec.detach().cpu().numpy()
            mse_vec, _ = site.perturb_and_measure(float(SIGMA_REF), src)
            resp_ref[src, :] = mse_vec.detach().cpu().numpy()
            if (src + 1) % 10 == 0:
                print(f"  sources {src + 1}/{C} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    elapsed = (time.time() - t0) / 60.0
    assert not np.isnan(mse_curve).any()

    np.savez_compressed(
        out_dir / f"subject_{args.subject_idx}.npz",
        mse_curve=mse_curve, mean_diff_curve=md_curve, resp_ref=resp_ref,
        grid=np.array(grid, dtype=np.float64), sigma_ref=np.float64(SIGMA_REF),
    )

    # ---- 快速诊断：动态范围 + 三个 q 的 interior 比例（σ* 经 walk 原语） ----
    mse = mse_curve.astype(np.float64)
    floor, ceil = mse[:, 0, :], mse[:, -1, :]
    dyn = ceil / np.maximum(floor, 1e-30)
    resp = resp_ref.astype(np.float64)
    diag = {
        "subject": f"s{sid}", "n_used": int(n_used), "elapsed_min": round(elapsed, 3),
        "noise_model": "unbiased_gaussian_mu0 (sigma*xi on z-scored x^p, before model)",
        "floor_q05/50/95": [float(np.quantile(floor, q)) for q in (0.05, 0.5, 0.95)],
        "ceil_q05/50/95": [float(np.quantile(ceil, q)) for q in (0.05, 0.5, 0.95)],
        "dyn_ratio_q50": float(np.quantile(dyn, 0.5)),
        "dyn_ratio_q05": float(np.quantile(dyn, 0.05)),
        "nonmono_pair_frac": float(np.mean(np.any(np.diff(mse, axis=1) < 0, axis=1))),
        "interior_by_tau_quantile": {},
    }
    grid_l = list(grid)
    for q in (0.25, 0.5, 0.75):
        tau = compute_global_tau(torch.from_numpy(resp_ref), q)
        S = C
        sig_mat = np.empty((S, C), np.float32)
        tau_vec = torch.full((C,), float(tau))
        curve_t = torch.from_numpy(mse_curve)
        for s in range(S):
            sig_mat[s] = walk_monotone_curve(curve_t[s], grid_l, tau_vec,
                                             SIGMA_MIN, SIGMA_MAX).numpy()
        v = sig_mat.flatten()
        at_min = float(np.mean(v <= SIGMA_MIN + 1e-6)) * 100
        at_max = float(np.mean(v >= SIGMA_MAX - 1e-6)) * 100
        diag["interior_by_tau_quantile"][f"q={q}"] = {
            "tau": float(tau),
            "pct_interior": 100 - at_min - at_max, "pct_at_min": at_min, "pct_at_max": at_max,
            "sigma_median": float(np.median(v)),
        }
        print(f"  tau(q={q})={tau:.3e}: interior={100 - at_min - at_max:.1f}% "
              f"@min={at_min:.1f}% @max={at_max:.1f}% med={np.median(v):.4f}", flush=True)
    print(f"  dyn_ratio q05/q50 = {diag['dyn_ratio_q05']:.1f}x / {diag['dyn_ratio_q50']:.1f}x",
          flush=True)
    with open(out_dir / f"subject_{args.subject_idx}.diagnostics.json", "w") as fh:
        json.dump(diag, fh, indent=2)

    # ---- σ-响应（MSE）曲线图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": C_INK2, "axes.labelcolor": C_INK, "text.color": C_INK,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    flat = mse.reshape(C * C, N_GRID)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for row in flat[::3]:
        ax.plot(grid, row, color=C_INK2, alpha=0.05, lw=0.5)
    med = np.median(flat, axis=0)
    q25, q75 = np.quantile(flat, [0.25, 0.75], axis=0)
    ax.fill_between(grid, q25, q75, color=C_SERIES1, alpha=0.15, lw=0, label="IQR across pairs")
    ax.plot(grid, med, color=C_SERIES1, lw=2.0, label="median response")
    ax.axvline(SIGMA_MIN, color=C_CLAMP, lw=1.2, ls=":")
    ax.axvline(SIGMA_MAX, color=C_CLAMP, lw=1.2, ls=":")
    ax.axvline(SIGMA_REF, color=C_INK2, lw=0.8, ls="--", alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(SIGMA_MIN, ymax, " σ_min=0.01", color=C_CLAMP, va="top", fontsize=9)
    ax.text(SIGMA_MAX, ymax, "σ_max=0.48 ", color=C_CLAMP, va="top", ha="right", fontsize=9)
    ax.text(SIGMA_REF, ymin, "σ_ref=0.3", color=C_INK2, fontsize=9, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("perturbation strength σ")
    ax.set_ylabel("per-target output MSE")
    ax.set_title(f"σ–MSE response, UNBIASED noise (μ=0, σ·ξ) — {sid}, "
                 f"{C}×{C} pairs, log grid [{SIGMA_MIN},{SIGMA_MAX}]×{N_GRID}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "response_curves_unbiased.png", dpi=160)
    print(f"done {elapsed:.2f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
