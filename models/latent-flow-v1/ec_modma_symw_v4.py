#!/usr/bin/env python3
"""ec_modma_symw_v4.py — MODMA fold0 × symw-exit-GNN ckpt 的 v4 无偏扰动 EC + GC 相关.

任务 10-28-modma-symw-f0。两模式：

- main（GPU）：镜像 ec_modma_globaltau_unbiased 主路径（无偏 μ=0、两阶段 global-τ、
  stage-2 每源单 autocast 域），07-05 家规参数（q=0.5 / σ_ref=0.3 /
  grid log[0.01,0.3]×16 / 每例 32 窗 / n_steps=10 / batch=32 / seed=42）；
  差异仅两处：① ckpt = train_modma_symw 产物（symw-exit-GNN，训练侧同一构建函数
  build_model_modma_symw 装载）；② conn 公式 = v4 交付公式
  −(ln σ* + 0.5·ln2πe) + 5/7（mild 线默认；与 07-05 原公式差常数 +5/7，
  秩相关完全一致，绝对值平移）。
- gccorr（离线）：main 产物 × 10-27 现算 GC（outputs/10-27-gc-fold0-aic-anchor/
  gc_aic/gc，granger_bivariate_spectral AIC max8/1-125Hz/nf65/fs250/全长，
  经旧库残留对拍逐位一致验证）；offdiag 16256 对，|EC|/raw × aligned/flipped ×
  Spearman/Pearson，逐被试 + 组均值矩阵（协议对齐 mild 线 run_gccorr 先例）。

Run:
  OMP_NUM_THREADS=8 MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 \
    models/latent-flow-v1/ec_modma_symw_v4.py --mode main \
    --ckpt .../fold0_modma_symw_exitgnn_graph_mexhat.pt \
    --out-dir /root/npi/outputs/10-28-modma-symw-f0/ec_main
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))

from latent_flow.attribution.bisect_sigma import (  # noqa: E402
    SignalProbeSite,
    compute_global_tau,
    log_grid,
    sigma_star_with_global_tau,
)
from latent_flow.scripts._connectivity_driver import (  # noqa: E402
    get_fold_test_subjects,
    load_subject_trials,
    subsample_trials,
)
from latent_flow.training.seeding import set_seed  # noqa: E402
from train_modma_symw import build_model_modma_symw  # noqa: E402

# --- 07-05 家规参数（MODMA 线口径；与 mild 线 grid[0.01,0.48]×24 不同，勿混） ---
DEFAULT_Q = 0.5
DEFAULT_SIGMA_REF = 0.3
SIGMA_MIN, SIGMA_MAX, DEFAULT_N_GRID = 0.01, 0.3, 16
DEFAULT_N_TRIALS = 32
DEFAULT_SEED = 42
N_STEPS, BATCH = 10, 32
DATASET = "modma"
HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)
V4_OFFSET = 5.0 / 7.0
GC_DIR = Path("/root/npi/outputs/10-27-gc-fold0-aic-anchor/gc_aic/gc")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", required=True, choices=["main", "gccorr"])
    p.add_argument("--arch", choices=["exitgnn-symw", "std-mexhat"],
                   default="exitgnn-symw",
                   help="模型构建（默认 exitgnn-symw = 旧行为；std-mexhat = "
                        "07-05 口径标准构建，10-29-modma-0705-f1234 起）")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--ckpt", type=str,
                   default="/root/npi/outputs/10-28-modma-symw-f0/run_01/"
                           "checkpoints/fold0_modma_symw_exitgnn_graph_mexhat.pt")
    p.add_argument("--subjects", type=int, default=10,
                   help="主跑被试数（fold0 test 共 10；冒烟可传 1）")
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--sources", type=int, default=0,
                   help="扰动源通道数（默认全 128；冒烟可传 8）")
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--sigma-ref", type=float, default=DEFAULT_SIGMA_REF)
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=SIGMA_MAX)
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True,
                   help="输出目录（必填，无默认——防误写旧任务树）")
    p.add_argument("--main-dir", type=str, default="",
                   help="gccorr 模式：subject_*.json 所在目录（默认 = --out-dir）")
    p.add_argument("--gc-dir", type=str, default=str(GC_DIR),
                   help="gccorr 模式：10-27 GC json 目录")
    p.add_argument("--overwrite", action="store_true",
                   help="main 模式：json 已存在也重算（默认跳过=断点续跑）")
    return p.parse_args(argv)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", "/root/npi", "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unavailable"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _load_ckpt(model, ckpt_path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = (payload["state_dict"] if isinstance(payload, dict)
             and payload.get("format_version") == 1 else payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)


def run_main(a) -> None:
    device = torch.device(a.device if a.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_channels = 128
    if a.sources <= 0:
        a.sources = n_channels
    assert Path(a.ckpt).exists(), f"ckpt not found: {a.ckpt}"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(a.seed)
    if a.arch == "std-mexhat":
        from train_modma_0705 import build_model_modma_std
        model = build_model_modma_std(device)
    else:
        model = build_model_modma_symw(device)
    _load_ckpt(model, a.ckpt)
    model.eval()

    test_files = get_fold_test_subjects(a.fold, a.seed, dataset=DATASET)
    subjects_to_run = test_files[a.subject_idx:a.subject_idx + a.subjects]
    assert subjects_to_run, "--subjects 切片为空"

    grid = log_grid(a.sigma_min, a.sigma_max, a.n_grid)
    print(f"[main] paradigm: global-quantile-tau UNBIASED mu=0 (07-05 params) | "
          f"q={a.q} sigma_ref={a.sigma_ref} grid=log[{a.sigma_min},{a.sigma_max}]"
          f"x{a.n_grid} | conn v4 | fold={a.fold} subjects={len(subjects_to_run)} "
          f"n_trials={a.n_trials} | device={device}", flush=True)

    enabled = device.type == "cuda"
    t_all = time.time()
    for i, subject_file in enumerate(subjects_to_run):
        subj_idx = a.subject_idx + i
        json_path = out_dir / f"subject_{subj_idx}.json"
        if json_path.exists() and not a.overwrite:
            print(f"[main] subject_{subj_idx} json 已存在，跳过（断点续跑）",
                  flush=True)
            continue
        t0 = time.time()
        x_obs_full = load_subject_trials(subject_file, device, normalize=True,
                                         dataset=DATASET)
        x_obs, n_used = subsample_trials(x_obs_full, a.n_trials, a.seed, subj_idx)
        site = SignalProbeSite(
            model=model, x_obs_subject=x_obs,
            n_steps=a.n_steps, use_proj=False,
            batch_size=a.batch_size, seed=a.seed,
            n_grid=a.n_grid,
            sigma_min=a.sigma_min, sigma_max=a.sigma_max,
            use_biased_noise=False,
        )
        assert site._mu_bias is None, "μ_bias 未关闭（无偏契约破坏）"

        n_failed = 0
        try:
            # stage 1：σ_ref 响应扫描 → τ_global（flatten 含对角自响应，先例同口径）
            response = np.full((a.sources, n_channels), np.nan, dtype=np.float32)
            for src in range(a.sources):
                site.set_source(src)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=enabled):
                    mse_vec = site.perturb_and_measure(float(a.sigma_ref), src)
                    if isinstance(mse_vec, tuple):
                        mse_vec = mse_vec[0]
                response[src, :] = mse_vec.detach().float().cpu().numpy()
            tau_global = compute_global_tau(torch.from_numpy(response), a.q)

            # stage 2：log 网格走查 → σ*（每源单 autocast 域）
            sigma_matrix = np.full((a.sources, n_channels), np.nan,
                                   dtype=np.float32)
            for src in range(a.sources):
                site.set_source(src)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=enabled):
                    mse_curve = torch.stack(
                        [site.perturb_and_measure(float(s), src)[0]
                         for s in grid], dim=0)
                sigma_matrix[src, :] = sigma_star_with_global_tau(
                    mse_curve, grid, float(tau_global),
                    a.sigma_min, a.sigma_max).numpy()
        except RuntimeError as exc:
            print(f"  FAILED subject {subj_idx}: {type(exc).__name__}: {exc}",
                  flush=True)
            n_failed = a.sources
            sigma_matrix = np.full((a.sources, n_channels), np.nan,
                                   dtype=np.float32)
            response = np.zeros((a.sources, n_channels), dtype=np.float32)
            tau_global = float("nan")

        # v4 conn：−(ln σ* + ½ln2πe) + 5/7（与 07-05 原公式差常数，秩不变）
        sigma_t = torch.from_numpy(sigma_matrix.copy())
        conn = -(sigma_t.float().log() + HALF_LOG_2PIE).numpy() + V4_OFFSET
        v = sigma_matrix.flatten()
        n_min = int(np.sum(np.isfinite(v) & (v <= a.sigma_min + 1e-6)))
        n_max = int(np.sum(np.isfinite(v) & (v >= a.sigma_max - 1e-6)))
        p_resp = float(np.mean(response.flatten() > tau_global)) \
            if np.isfinite(tau_global) else float("nan")
        elapsed = (time.time() - t0) / 60.0
        payload = {
            "task": "10-28-modma-symw-f0",
            "subject_idx": int(subj_idx),
            "subject_file": Path(subject_file).name,
            "n_trials": int(n_used),
            "n_sources_requested": int(a.sources),
            "n_sources_succeeded": int(a.sources - n_failed),
            "n_source_failures": int(n_failed),
            "method": ("global_tau_quantile UNBIASED (mu=0, sigma*xi, "
                       f"latent-flow MODMA {a.arch} surrogate, "
                       "signal-space, sigma* preserved)"),
            "tau_global": float(tau_global),
            "q": float(a.q),
            "sigma_ref": float(a.sigma_ref),
            "sigma_matrix": sigma_matrix.tolist(),
            "connectivity_matrix": conn.tolist(),
            "connectivity_formula":
                "-(log sigma* + 0.5*log(2*pi*e)) + 5/7 (v4; constant shift of "
                "07-05 form, rank-invariant)",
            "raw_response_matrix": response.tolist(),
            "noise_model": ("unbiased_gaussian_mu0 (sigma*xi additive on "
                            "z-scored x_obs, before model; same xi across "
                            "sigma per source)"),
            "grid": "log_spaced",
            "hparams": {"q": float(a.q), "sigma_ref": float(a.sigma_ref),
                        "sigma_min": float(a.sigma_min),
                        "sigma_max": float(a.sigma_max),
                        "n_grid": int(a.n_grid), "n_steps": int(a.n_steps),
                        "batch_size": int(a.batch_size), "seed": int(a.seed),
                        "use_biased_noise": False, "fold": int(a.fold),
                        "arch": a.arch},
            "checkpoint": Path(a.ckpt).name,
            "git_head": _git_head(),
            "diagnostics": {
                "pct_at_sigma_min": n_min / v.size * 100,
                "pct_at_sigma_max": n_max / v.size * 100,
                "pct_interior": 100 - (n_min + n_max) / v.size * 100,
                "p_response_above_tau": p_resp,
            },
            "elapsed_min": elapsed,
        }
        _atomic_write_json(json_path, payload)
        print(f"[main] subject_{subj_idx} ({Path(subject_file).name}) "
              f"{elapsed:.2f} min | tau={tau_global:.3e} "
              f"interior={payload['diagnostics']['pct_interior']:.1f}% "
              f"@min={payload['diagnostics']['pct_at_sigma_min']:.1f}% "
              f"@max={payload['diagnostics']['pct_at_sigma_max']:.1f}% "
              f"P(resp>tau)={p_resp:.3f} | wrote {json_path.name}", flush=True)
    print(f"[main] done in {(time.time()-t_all)/60.0:.1f} min -> {out_dir}",
          flush=True)


def run_gccorr(a) -> None:
    from scipy import stats as sps
    main_dir = Path(a.main_dir or a.out_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gc = {}
    for f in sorted(Path(a.gc_dir).glob("subject_*.json")):
        d = json.load(open(f))
        gc[d["subject_file"]] = np.asarray(d["connectivity_matrix"],
                                           dtype=np.float64)
    assert gc, f"GC 目录为空: {a.gc_dir}"
    off = ~np.eye(128, dtype=bool)

    per, stack_conn, stack_gc = [], [], []
    for f in sorted(main_dir.glob("subject_*.json")):
        d = json.load(open(f))
        name = d["subject_file"]
        assert name in gc, f"GC 缺 {name}"
        conn = np.asarray(d["connectivity_matrix"], dtype=np.float64)
        g = gc[name]
        r = {"subject": name, "tau": d.get("tau_global")}
        for tag, mat in (("entropy", conn),
                         ("inv", 1.0 / (np.asarray(d["sigma_matrix"],
                                                   dtype=np.float64) + 1e-4))):
            A, R = np.abs(mat), mat
            r[f"spearman_abs_aligned_{tag}"] = float(
                sps.spearmanr(A[off], g[off]).statistic)
            r[f"pearson_abs_aligned_{tag}"] = float(
                np.corrcoef(A[off], g[off])[0, 1])
            r[f"spearman_raw_aligned_{tag}"] = float(
                sps.spearmanr(R[off], g[off]).statistic)
            r[f"spearman_abs_flipped_{tag}"] = float(
                sps.spearmanr(A[off], g.T[off]).statistic)
        per.append(r)
        stack_conn.append(conn)
        stack_gc.append(g)
        print(f"[gccorr] {name}: abs rho={r['spearman_abs_aligned_entropy']:+.4f} "
              f"r={r['pearson_abs_aligned_entropy']:+.4f} "
              f"(flip {r['spearman_abs_flipped_entropy']:+.4f}; "
              f"inv rho={r['spearman_abs_aligned_inv']:+.4f})", flush=True)

    def agg(k):
        v = np.array([p[k] for p in per], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    A_mean = np.stack([np.abs(c) for c in stack_conn]).mean(axis=0)
    gc_mean = np.stack(stack_gc).mean(axis=0)
    summary = {
        "task": "10-28-modma-symw-f0 Run gccorr (descriptive EC x GC)",
        "naming": ("EC_pert = unbiased mu=0 global_tau (q=0.5@sigma_ref=0.3, "
                   "grid log[0.01,0.3]x16) conn v4 on latent-flow MODMA "
                   "symw-exit-gnn surrogate; GC = granger_bivariate_spectral "
                   "AIC max8/1-125Hz/nf65/fs250 full-length (10-27 recomputed, "
                   "bitwise-validated vs old lib)"),
        "protocol": ("offdiag 16256 pairs; aligned = row=source vs gc; "
                     "flipped = vs gc.T; entropy v4 all-positive => abs==raw; "
                     "inv = 1/(sigma*+1e-4) (Spearman identical to entropy)"),
        "descriptive_only": ("描述性并置登记（先例协议），不构成代理脑资格判定；"
                             "EC=代理脑内部量 vs GC=观测数据量，实现口径隔离"),
        "per_subject": per,
        "group": {k: agg(k) for k in per[0]
                  if k.startswith(("spearman", "pearson"))},
        "group_mean": {
            "spearman_abs_aligned": float(
                sps.spearmanr(A_mean[off], gc_mean[off]).statistic),
            "pearson_abs_aligned": float(
                np.corrcoef(A_mean[off], gc_mean[off])[0, 1])},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_write_json(out_dir / "gccorr_summary.json", summary)
    g = summary["group"]
    print(f"\n[gccorr] GROUP abs aligned (entropy): "
          f"rho {g['spearman_abs_aligned_entropy']['mean']:+.4f}"
          f"±{g['spearman_abs_aligned_entropy']['std']:.4f} | "
          f"r {g['pearson_abs_aligned_entropy']['mean']:+.4f}"
          f"±{g['pearson_abs_aligned_entropy']['std']:.4f} | "
          f"group_mean rho={summary['group_mean']['spearman_abs_aligned']:+.4f} "
          f"-> {out_dir / 'gccorr_summary.json'}", flush=True)


def main() -> int:
    a = parse_args()
    if a.mode == "main":
        run_main(a)
    else:
        run_gccorr(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
