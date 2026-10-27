#!/usr/bin/env python3
"""ec_modma_globaltau_unbiased.py — 07-05 口径无偏 global-τ 复现（MODMA fold0 × graph_mexhat）.

任务 10-27-lf-globaltau-f0-unbiased-repro（PRD 契约实现）。复现目标 = 旧库
07-04-global-tau-sigma 任务的 2026-07-05 运行：fold0 `graph_mexhat` 原版 ckpt +
无偏零均值噪声 ξ~N(0,1) + 两阶段 global-τ σ*（信号空间），q=0.5 / σ_ref=0.3 /
σ∈log[0.01,0.3]×**16** / n_steps=10 / batch=32 / seed=42 / fold0 test 10 例 × 32 窗。
原始逐被试产物已被清理（results/oc_connectivity_global_tau_signal_q0.5 不在旧库快照），
本复现的可核对锚点 = domain.md §07-05 仅存聚合数：OC↔SC +0.31、行方差 5.7%、
残差方差 65.5%、vs A_eff +0.36（GC/TE 锚点因 fold0 基线矩阵本机缺失不核，PRD 注明）。

与移植版 runner 的差异（仅此三处，其余逐行同口径）：
  1) SignalProbeSite(use_biased_noise=False)（runner 硬编码 True，07-23 后形态；
     07-05 早于有偏噪声存在）；
  2) method/JSON 键按 07-05 原版契约：method=global_tau_quantile，无
     sign/mean_diff/mu_bias 字段；
  3) conn 公式 = −(ln σ* + ½·ln2πe)（无 +5/7 平移——v4 为 mild 线决策，MODMA 旧线
     保持原式）。

复用（零拷贝，包体零改动）：bisect_sigma.SignalProbeSite / log_grid /
compute_global_tau / sigma_star_with_global_tau / _HALF_LOG_2PIE；
_connectivity_driver.load_model / get_fold_test_subjects / load_subject_trials /
subsample_trials / sanitize_for_json；extract_graph_connectivity.
effective_connectivity（A_eff 重建锚点）。

两模式：main（GPU 主跑，逐例原子写 JSON，已存在即跳过=断点续跑）；
anchors（离线：钳边/方差分解聚合 + OC↔SC Spearman + vs A_eff 并置表，
对照 domain.md §07-05 原始值，跨栈 caveat 内嵌）。

Run:
  OMP_NUM_THREADS=8 python3 models/latent-flow-v1/ec_modma_globaltau_unbiased.py \
    --mode main --out-dir /root/npi/outputs/10-27-lf-globaltau-f0-unbiased-repro/run_01
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境契约） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))

from latent_flow.attribution.bisect_sigma import (  # noqa: E402
    SIGMA_MAX,
    SIGMA_MIN,
    SignalProbeSite,
    _HALF_LOG_2PIE,
    compute_global_tau,
    log_grid,
    sigma_star_with_global_tau,
)
from latent_flow.data.datasets import get_dataset  # noqa: E402
from latent_flow.scripts._connectivity_driver import (  # noqa: E402
    get_fold_test_subjects,
    json_default,
    load_model,
    load_subject_trials,
    sanitize_for_json,
    subsample_trials,
)
from latent_flow.scripts.extract_graph_connectivity import (  # noqa: E402
    effective_connectivity,
)

# --- 07-05 原始口径常量（domain.md §07-05：q=0.5 / n_grid=16 / n_trials=32 / seed=42） ---
DEFAULT_Q = 0.5
DEFAULT_SIGMA_REF = 0.3
DEFAULT_N_GRID = 16          # 原始运行实测值（非脚本默认 24）——复现契约
DEFAULT_N_TRIALS = 32
DEFAULT_SEED = 42
DEFAULT_FOLD = 0
N_STEPS, BATCH = 10, 32
DATASET = "modma"            # 本复现仅 MODMA（128ch）
DEFAULT_CKPT = ("/root/autodl-tmp/latent_ode/checkpoints/"
                "fold0_oc_latent_flow_flow_best_graph_mexhat.pt")
SC_PRIOR_PT = ("/root/autodl-tmp/latent_ode/results/sc_prior/"
               "sc_elec_directed.pt")
# domain.md §07-05 仅存原始聚合值（9 例验证集；原始逐被试产物已被清理）——并置对照常量
ANCHORS_0705 = {
    "oc_vs_sc_spearman": +0.31,
    "row_source_variance_pct": 5.7,
    "residual_edge_variance_pct": 65.5,
    "signal_vs_aeff_spearman": +0.36,
    "oc_vs_gc_spearman": +0.29,   # 本机无 fold0 GC 基线矩阵，不核（PRD 注明）
    "oc_vs_te_spearman": +0.28,   # 同上
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=["main", "anchors"])
    p.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--subjects", type=int, default=10,
                   help="主跑被试数（fold0 test 共 10；冒烟可传 1）")
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--sources", type=int, default=0,
                   help="扰动源通道数（默认全 128；冒烟可传 16）")
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
                   help="anchors 模式：subject_*.json 所在目录（默认 = --out-dir）")
    p.add_argument("--overwrite", action="store_true",
                   help="main 模式：json 已存在也重算（默认跳过=断点续跑）")
    p.add_argument("--sc-prior", type=str, default=SC_PRIOR_PT,
                   help="anchors 模式：SC 电极空间有向先验 .pt 路径")
    return p.parse_args(argv)


def _caliber_asserts(a) -> None:
    assert a.fold == DEFAULT_FOLD, (
        f"07-05 复现口径钉死 --fold {DEFAULT_FOLD}（原始运行折），got fold={a.fold}")
    assert (a.q, a.n_grid, a.n_trials, a.seed) == (DEFAULT_Q, DEFAULT_N_GRID,
                                                   DEFAULT_N_TRIALS, DEFAULT_SEED), (
        "07-05 复现口径钉死 q=0.5/n_grid=16/n_trials=32/seed=42（改口径须显式"
        "偏离登记，勿动默认）")
    assert Path(a.ckpt).exists(), f"ckpt not found: {a.ckpt}"
    assert Path(a.ckpt).name == "fold0_oc_latent_flow_flow_best_graph_mexhat.pt", (
        "复现契约 = graph_mexhat 原版 ckpt（07-04 PRD R6 点名），got "
        f"{Path(a.ckpt).name}")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
    os.replace(tmp, path)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd="/root/npi").stdout.strip()
    except Exception:
        return "unavailable"


# =============================================================================
# main 模式：两阶段 global-τ σ*（无偏 μ=0，GPU）
# =============================================================================

def run_main(a) -> None:
    _caliber_asserts(a)
    device = torch.device(a.device if a.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_channels = get_dataset(DATASET).channels
    if a.sources <= 0:
        a.sources = n_channels
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(a.ckpt, device)   # _load_model_compat 直通（07-05 pre-SC ckpt 兼容路径）
    test_files = get_fold_test_subjects(a.fold, a.seed, dataset=DATASET)
    subjects_to_run = test_files[a.subject_idx:a.subject_idx + a.subjects]
    assert subjects_to_run, "--subjects 切片为空"

    grid = log_grid(a.sigma_min, a.sigma_max, a.n_grid)
    print(f"[main] paradigm: global-quantile-tau UNBIASED mu=0 (07-05 caliber) | "
          f"q={a.q} sigma_ref={a.sigma_ref} grid=log[{a.sigma_min},{a.sigma_max}]"
          f"x{a.n_grid} | fold={a.fold} subjects={len(subjects_to_run)} "
          f"n_trials={a.n_trials} | device={device}", flush=True)

    t_all = time.time()
    for i, subject_file in enumerate(subjects_to_run):
        subj_idx = a.subject_idx + i
        json_path = out_dir / f"subject_{subj_idx}.json"
        if json_path.exists() and not a.overwrite:
            print(f"[main] subject_{subj_idx} json 已存在，跳过（断点续跑）", flush=True)
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

        enabled = device.type == "cuda"
        n_failed = 0
        try:
            # stage 1：σ_ref 响应扫描 → τ_global（flatten 含对角自响应，runner 同口径）
            response = np.full((a.sources, n_channels), np.nan, dtype=np.float32)
            for src in range(a.sources):
                site.set_source(src)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=enabled):
                    mse_vec = site.perturb_and_measure(float(a.sigma_ref), src)
                    if isinstance(mse_vec, tuple):
                        mse_vec = mse_vec[0]   # 无偏契约：只取 MSE，mean_diff 不落档
                response[src, :] = mse_vec.detach().float().cpu().numpy()
            tau_global = compute_global_tau(torch.from_numpy(response), a.q)
            print(f"  stage 1: tau(q={a.q}) = {tau_global:.6e}", flush=True)

            # stage 2：log 网格走查 → σ*（每源单 autocast 域，runner _stage2 同构）
            sigma_matrix = np.full((a.sources, n_channels), np.nan,
                                   dtype=np.float32)
            for src in range(a.sources):
                site.set_source(src)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=enabled):
                    mse_curve = torch.stack(
                        [site.perturb_and_measure(float(s), src)[0] for s in grid],
                        dim=0)
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

        sigma_t = torch.from_numpy(sigma_matrix.copy())
        conn = -(sigma_t.float().log() + _HALF_LOG_2PIE).numpy()
        v = sigma_matrix.flatten()
        n_min = int(np.sum(np.isfinite(v) & (v <= a.sigma_min + 1e-6)))
        n_max = int(np.sum(np.isfinite(v) & (v >= a.sigma_max - 1e-6)))
        elapsed = (time.time() - t0) / 60.0
        payload = {
            "subject_idx": int(subj_idx),
            "subject_file": Path(subject_file).name,
            "n_trials": int(n_used),
            "n_sources_requested": int(a.sources),
            "n_sources_succeeded": int(a.sources - n_failed),
            "n_source_failures": int(n_failed),
            "method": "global_tau_quantile (signal-space, sigma* preserved)",
            "tau_global": float(tau_global),
            "tau": float(tau_global),   # 07-05 原版键名（implement.md §2.3 契约）
            "q": float(a.q),
            "sigma_ref": float(a.sigma_ref),
            "sigma_matrix": sigma_matrix.tolist(),
            "connectivity_matrix": conn.tolist(),
            "connectivity_formula": "-(log sigma* + 0.5*log(2*pi*e))",
            "raw_response_matrix": response.tolist(),
            "noise_model": ("unbiased_gaussian_mu0 (sigma*xi additive on z-scored "
                            "x_obs, before model; same xi across sigma per source)"),
            "grid": "log_spaced",
            "hparams": {"q": float(a.q), "sigma_ref": float(a.sigma_ref),
                        "sigma_min": float(a.sigma_min),
                        "sigma_max": float(a.sigma_max),
                        "n_grid": int(a.n_grid), "n_steps": int(a.n_steps),
                        "batch_size": int(a.batch_size), "seed": int(a.seed),
                        "use_biased_noise": False, "fold": int(a.fold)},
            "checkpoint": Path(a.ckpt).name,
            "reproduction": {
                "target": "old-lib task 07-04-global-tau-sigma run of 2026-07-05 "
                          "(fold0, graph_mexhat ckpt, unbiased mu=0, q=0.5, "
                          "n_grid=16, n_trials=32, seed=42)",
                "original_outputs_status": "deleted from old-lib snapshot "
                                           "(results/oc_connectivity_global_tau_"
                                           "signal_q0.5 absent) — anchors from "
                                           "domain.md sec-07-05 only",
                "cross_stack_caveat": "original Windows/Python3.13/torch2.12/"
                                      "5070Ti vs this Linux/Python3.12/torch2.8/"
                                      "4080S — same seed & code path, statistical-"
                                      "level agreement expected, bit-identity NOT "
                                      "guaranteed"},
            "diagnostics": {
                "n_at_sigma_min": n_min, "n_at_sigma_max": n_max,
                "pct_at_sigma_min": n_min / v.size * 100,
                "pct_at_sigma_max": n_max / v.size * 100,
                "pct_interior": 100 - (n_min + n_max) / v.size * 100,
            },
            "elapsed_min": float(elapsed),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_head": _git_head(),
        }
        payload = sanitize_for_json(payload)
        _atomic_write_json(json_path, payload)
        print(f"[main] subject_{subj_idx} ({Path(subject_file).name}) "
              f"{elapsed:.2f} min | tau={tau_global:.6e} "
              f"@min={n_min / v.size * 100:.1f}% @max={n_max / v.size * 100:.1f}%"
              f" | wrote {json_path.name}", flush=True)
    print(f"[main] done in {(time.time() - t_all) / 60:.1f} min -> {out_dir}",
          flush=True)


# =============================================================================
# anchors 模式：离线锚点核对并置（无 GPU 前向；A_eff 重建仅 ckpt 常量前向）
# =============================================================================

def _variance_decomposition(m: np.ndarray, n_iter: int = 10) -> dict:
    """offdiag 方差的行(源)/列(靶)/残差占比（交替均值消主效应；diag 置 nan 屏蔽）。

    与 ec_mild_unbiased._variance_decomposition 同款（旧库 diag 口径的移植形态）；
    domain.md §07-05 的 5.7%/65.5% 为旧库 diag_de_vs_baseline 产出，方法为同型
    行/列/残差分解， recipe 假设随档注明（统计级对照，不逐位对拍）。
    """
    a = m.astype(np.float64).copy()
    np.fill_diagonal(a, np.nan)
    grand = np.nanmean(a)
    row_eff = np.nanmean(a, axis=1) - grand
    col_eff = np.nanmean(a, axis=0) - grand
    resid = a - grand - row_eff[:, None] - col_eff[None, :]
    for _ in range(n_iter):
        resid -= np.nanmean(resid, axis=1, keepdims=True)
        resid -= np.nanmean(resid, axis=0, keepdims=True)
        row_eff = np.nanmean(a - resid, axis=1) - grand
        col_eff = np.nanmean(a - resid, axis=0) - grand
    v_row = np.nanvar(grand + row_eff[:, None] * np.ones_like(a))
    v_col = np.nanvar(grand + col_eff[None, :] * np.ones_like(a))
    v_res = np.nanvar(resid)
    tot = v_row + v_col + v_res
    return {"row_source_share": v_row / tot, "col_target_share": v_col / tot,
            "residual_edge_share": v_res / tot}


def _load_sc(prior_pt: str) -> np.ndarray:
    obj = torch.load(prior_pt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("sc_elec", "sc", "matrix", "sc_elec_directed"):
            if k in obj and hasattr(obj[k], "numpy"):
                obj = obj[k]
                break
    mat = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], f"SC 形状异常 {mat.shape}"
    return mat


def run_anchors(a) -> None:
    from scipy import stats as sps
    main_dir = Path(a.main_dir or a.out_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(main_dir.glob("subject_*.json"))
    assert rows, f"no subject_*.json under {main_dir}"
    n_ch = get_dataset(DATASET).channels
    off = ~np.eye(n_ch, dtype=bool)

    device = torch.device("cpu")   # A_eff 重建 = ckpt 常量前向，CPU 即可
    model = load_model(a.ckpt, device)
    aeff = effective_connectivity(model).astype(np.float64)
    assert aeff.shape == (n_ch, n_ch), f"A_eff 形状异常 {aeff.shape}"
    sc = _load_sc(a.sc_prior)
    if sc.shape[0] != n_ch:
        raise AssertionError(f"SC 通道数 {sc.shape[0]} != MODMA {n_ch}（07-05 口径 "
                             "SC_elec 为 128 电极空间，请核路径）")

    per, conn_stack = [], []
    for f in rows:
        d = json.load(open(f))
        conn = np.asarray(d["connectivity_matrix"], dtype=np.float64)
        sig = np.asarray(d["sigma_matrix"], dtype=np.float64)
        vd = _variance_decomposition(conn)
        r = {
            "subject": d["subject_file"], "tau": d["tau_global"],
            "pct_at_min": d["diagnostics"]["pct_at_sigma_min"],
            "pct_at_max": d["diagnostics"]["pct_at_sigma_max"],
            "pct_interior": d["diagnostics"]["pct_interior"],
            "row_source_share": vd["row_source_share"],
            "col_target_share": vd["col_target_share"],
            "residual_edge_share": vd["residual_edge_share"],
            # SC（有向，行=源）：raw 与 abs 两种读出并记（domain.md 未注明 abs 与否，
            # conn 无符号强度近全正，两者应近同——recipe 假设随档）
            "spearman_sc_raw_aligned": float(sps.spearmanr(
                conn[off], sc[off]).statistic),
            "spearman_sc_abs_aligned": float(sps.spearmanr(
                np.abs(conn)[off], np.abs(sc)[off]).statistic),
            "spearman_sc_raw_flipped": float(sps.spearmanr(
                conn[off], sc.T[off]).statistic),
            # A_eff（重建，mexhat+γ·W 读出）
            "spearman_aeff_raw": float(sps.spearmanr(
                conn[off], aeff[off]).statistic),
            "spearman_aeff_abs": float(sps.spearmanr(
                np.abs(conn)[off], np.abs(aeff)[off]).statistic),
        }
        per.append(r)
        conn_stack.append(conn)

    def agg(key):
        v = np.array([p[key] for p in per], dtype=np.float64)
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "n": len(v)}

    conn_mean = np.stack(conn_stack).mean(axis=0)
    summary = {
        "task": "10-27-lf-globaltau-f0-unbiased-repro anchors "
                "(offline juxtaposition vs domain.md sec-07-05)",
        "n_subjects": len(per),
        "original_run": {
            "date": "2026-07-05", "validated_set": "9/10 (subject_9 hung, killed)",
            "ckpt": "fold0_oc_latent_flow_flow_best_graph_mexhat.pt",
            "caliber": "unbiased mu=0, q=0.5, n_grid=16, n_trials=32, seed=42"},
        "anchors_domain_md": ANCHORS_0705,
        "juxtaposition": [
            {"metric": "OC vs SC Spearman (raw, aligned)",
             "original_0705": ANCHORS_0705["oc_vs_sc_spearman"],
             "repro_mean": agg("spearman_sc_raw_aligned")},
            {"metric": "row (source) variance %",
             "original_0705": ANCHORS_0705["row_source_variance_pct"],
             "repro_mean_pct": agg("row_source_share")["mean"] * 100},
            {"metric": "residual edge variance %",
             "original_0705": ANCHORS_0705["residual_edge_variance_pct"],
             "repro_mean_pct": agg("residual_edge_share")["mean"] * 100},
            {"metric": "signal-space vs A_eff Spearman (rebuilt)",
             "original_0705": ANCHORS_0705["signal_vs_aeff_spearman"],
             "repro_mean": agg("spearman_aeff_raw")},
            {"metric": "OC vs GC Spearman",
             "original_0705": ANCHORS_0705["oc_vs_gc_spearman"],
             "repro_mean": None,
             "note": "NOT RECOMPUTABLE on this machine (fold0 per-subject GC "
                     "baseline matrices deleted; only PNGs remain)"},
            {"metric": "OC vs TE Spearman",
             "original_0705": ANCHORS_0705["oc_vs_te_spearman"],
             "repro_mean": None,
             "note": "NOT RECOMPUTABLE (same as GC)"},
        ],
        "per_subject": per,
        "group": {k: agg(k) for k in per[0] if k != "subject"},
        "group_mean_matrix_spearman": {
            "vs_sc": float(sps.spearmanr(conn_mean[off], sc[off]).statistic),
            "vs_aeff": float(sps.spearmanr(conn_mean[off], aeff[off]).statistic)},
        "recipe_assumptions": [
            "variance decomposition = alternating row/col mean removal "
            "(house port of old-lib diag caliber; original numbers produced by "
            "diag_de_vs_baseline.py — statistical-level juxtaposition, not "
            "bit-level)",
            "SC readout: raw + abs + flipped recorded; original did not record "
            "abs/flip variant — aligned raw is the primary comparison",
            "A_eff rebuilt from ckpt graph_mixer via effective_connectivity() "
            "(original A_eff json deleted from snapshot)",
            "cross-stack caveat: statistical-level agreement expected "
            "(Windows/torch2.12/5070Ti vs Linux/torch2.8/4080S)",
        ],
        "aeff_diag_note": "A_eff off-diag mean "
                          f"{float(aeff[~np.eye(n_ch, dtype=bool)].mean()):+.4f}",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "anchors_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    np.savez_compressed(out_dir / "aeff_rebuilt.npz", aeff=aeff)
    j = summary["juxtaposition"]
    print(f"[anchors] n={len(per)} | " +
          " | ".join(f"{x['metric']}: 0705={x['original_0705']} "
                     f"repro={x.get('repro_mean', x.get('repro_mean_pct'))}"
                     for x in j) +
          f" -> {out_dir / 'anchors_summary.json'}", flush=True)


def main() -> int:
    a = parse_args()
    if a.mode == "main":
        run_main(a)
    else:
        run_anchors(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
