"""τ 阈值搜索 — 10-21-ec-globaltau-v5main/threshold_search（2026-10-21）.

背景：run_01 (q=0.5, σ[0.01,0.3]) 边界钳制 95.8%，run_02 (q=0.75, σ[0.01,0.48]) 钳制
~92.5%——τ 取 σ_ref=0.3 响应的全局分位数导致大部分 (a,b) 对 σ* 被钳在 σ_min/σ_max。
本脚本在不改动扰动注入（σ·ξ+μ_bias，probe_site.py）与扰动数据构造（同 seeds/同子采样）
的前提下做阈值搜索：

  stage curves (GPU)：复用 GpfmProbeSite 逐源扫描全 σ 网格（log[0.01,0.48]×24，与 run_02
      同网格）+ σ_ref∈{0.3,0.1,0.2} 响应，落盘完整 MSE 曲线 (S,G,C) npz。τ 只在离线
      walk 中起作用 → 曲线一次扫描，全部候选阈值零额外前向。
  stage scan (CPU)：τ 候选 = 绝对值梯子 + 分位数族 (σ_ref,q) + 缩放族 c·τ(q=0.5@0.3)；
      σ* 数值一律经 import 的 walk_monotone_curve（禁止自实现）；钳制比例用与 walk 分支
      严格等价的向量化条件（mse[:,0]>τ → @min；全网格 mse≤τ → @max；否则内点）。
      产物：候选阈值统计表 CSV、逐被试最优 τ、扰动强度-响应曲线图、τ 扫描图、σ* ECDF、
      推荐配置 JSON。
  stage apply (CPU)：按推荐配置从已存曲线离线生成 subject JSON（σ*/conn/sign，run_02
      同公式）——前向确定性已由 run_01 S3（同 seed 逐位一致）保证，离线 σ* ≡ 重跑结果。

用法：
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      /root/miniconda3/bin/python models/ec-globaltau-v1/search_threshold.py \
        --stage curves --subjects 10 --n-trials 32
    /root/miniconda3/bin/python models/ec-globaltau-v1/search_threshold.py --stage scan
    /root/miniconda3/bin/python models/ec-globaltau-v1/search_threshold.py --stage apply
"""

import argparse
import json
import math
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

OUT_ROOT = Path("/root/npi/outputs/10-21-ec-globaltau-v5main/threshold_search")
RUN02_DIR = Path("/root/npi/outputs/10-21-ec-globaltau-v5main/run_02")

SIGMA_MIN, SIGMA_MAX, N_GRID = 0.01, 0.48, 24          # 与 run_02 完全一致
SIGMA_REFS = [0.3, 0.1, 0.2]                            # resp 列序；0.3 先行（对齐 run_02 stage1）
N_TRIALS, SEED = 32, 42
Q_LADDER = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
            0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
SCALE_LADDER = [0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.3, 5.0, 7.5, 11.0, 16.0, 24.0]
N_LADDER_DENSE, N_LADDER_WALK = 400, 120               # 稠密比例梯子 / σ* 走查子梯子

# dataviz 参考调色板（light）
C_SERIES1, C_SERIES2, C_SERIES3 = "#2a78d6", "#eb6834", "#1baf7a"
C_INK, C_INK2, C_GRID, C_SURFACE, C_CLAMP = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb", "#e34948"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["curves", "scan", "apply"], required=True)
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--subjects", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage A — GPU 曲线扫描（扰动注入/数据构造零改动，逐调用与 runner 同形态）
# ---------------------------------------------------------------------------

def stage_curves(args):
    from adapter import FOLD0_TEST_IDS, load_surrogate, subject_global_windows
    from probe_site import GpfmProbeSite

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out_dir = Path(args.out_root) / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = log_grid(SIGMA_MIN, SIGMA_MAX, N_GRID)
    cfg, model, ema, sampler, test_ds = load_surrogate(device=str(device))
    F = sampler.prior.future_len

    for i in range(args.subject_idx, args.subject_idx + args.subjects):
        npz_path = out_dir / f"subject_{i}.npz"
        if npz_path.exists():
            print(f"[curves] subject {i}: 已存在，跳过（可续跑）", flush=True)
            continue
        sid = FOLD0_TEST_IDS[i]
        widx = subject_global_windows(test_ds, sid)
        x_p_full = torch.stack([test_ds[gi][0] for gi in widx]).to(device)
        x_p, n_used = subsample_trials(x_p_full, args.n_trials, args.seed, i)
        N, C, P = x_p.shape
        print(f"[curves] subject {i} (s{sid}): windows {n_used} shape {tuple(x_p.shape)}", flush=True)

        # 生成噪声：被试级固定（与 runner 同式）
        gen = torch.Generator(device=device).manual_seed(args.seed * 10000 + sid)
        eps = torch.randn(N, 1, C, F, device=device, generator=gen)
        eta = torch.randn(N, 1, C, F, device=device, generator=gen)

        site = GpfmProbeSite(sampler, x_p, eps, eta, seed=args.seed)
        t0 = time.time()
        mse_curve = np.full((C, N_GRID, C), np.nan, dtype=np.float32)
        resp_ref = np.full((C, len(SIGMA_REFS), C), np.nan, dtype=np.float32)
        mean_diff_ref = np.full((C, C), np.nan, dtype=np.float32)
        with ema.swap(model):
            site.compute_baseline()
            for src in range(C):
                site.set_source(src)
                # σ_ref=0.3 先行（对齐 run_02 stage1 调用序），随后网格按索引，随后其余 σ_ref
                mse_vec, md_vec = site.perturb_and_measure(float(SIGMA_REFS[0]), src)
                resp_ref[src, 0, :] = mse_vec.detach().cpu().numpy()
                mean_diff_ref[src, :] = md_vec.detach().cpu().numpy()
                for gi, sig in enumerate(grid):
                    mse_vec, _ = site.perturb_and_measure(float(sig), src)
                    mse_curve[src, gi, :] = mse_vec.detach().cpu().numpy()
                for ri in (1, 2):
                    mse_vec, _ = site.perturb_and_measure(float(SIGMA_REFS[ri]), src)
                    resp_ref[src, ri, :] = mse_vec.detach().cpu().numpy()
        elapsed = (time.time() - t0) / 60.0
        assert not np.isnan(mse_curve).any(), "MSE 曲线存在 NaN"
        np.savez_compressed(
            npz_path,
            mse_curve=mse_curve, resp_ref=resp_ref, mean_diff_ref=mean_diff_ref,
            grid=np.array(grid, dtype=np.float64),
            sigma_refs=np.array(SIGMA_REFS, dtype=np.float64),
            mu_bias=(site._mu_bias.detach().cpu().numpy() if site._mu_bias is not None
                     else np.array([])),
        )
        meta = {
            "subject_idx": i, "subject_file": f"s{sid}", "n_used": int(n_used),
            "grid": {"sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX, "n_grid": N_GRID},
            "sigma_refs_eval_order": SIGMA_REFS, "seed": args.seed,
            "n_trials_cap": args.n_trials, "euler_steps": int(cfg.euler_steps),
            "ema_weights": True, "amp": "bf16-autocast",
            "elapsed_min": round(elapsed, 3), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out_dir / f"subject_{i}.meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[curves] subject {i} (s{sid}) done {elapsed:.2f} min -> {npz_path}", flush=True)


# ---------------------------------------------------------------------------
# Stage B — 离线 τ 扫描 + 统计 + 绘图 + 推荐
# ---------------------------------------------------------------------------

def load_all(curves_dir, n_subjects):
    data = []
    for i in range(n_subjects):
        p = curves_dir / f"subject_{i}.npz"
        if not p.exists():
            break
        z = np.load(p)
        data.append({
            "i": i, "sid": json.load(open(curves_dir / f"subject_{i}.meta.json"))["subject_file"],
            "mse": z["mse_curve"].astype(np.float64),          # (S,G,C)
            "resp": z["resp_ref"].astype(np.float64),          # (S,R,C) 列序 = SIGMA_REFS
            "md": z["mean_diff_ref"].astype(np.float64),
            "grid": z["grid"], "mu_bias": z["mu_bias"],
        })
    return data


def frac_split(mse, tau):
    """与 walk_monotone_curve 分支严格等价的向量化钳制判定（不替代 σ* 数值计算）。"""
    at_min = mse[:, 0, :] > tau                    # walk: mse_b[0] > tau_b → σ_min
    at_max = mse.max(axis=1) <= tau                # walk: stable.all() → σ_max
    interior = ~at_min & ~at_max
    return at_min, at_max, interior


def walk_sigma_matrix(mse, grid, tau_scalar):
    """σ* 数值一律经 import 的 walk_monotone_curve（禁止自实现）。mse (S,G,C) → (S,C)。"""
    S, _, C = mse.shape
    out = np.empty((S, C), np.float32)
    tau_vec = torch.full((C,), float(tau_scalar))
    curve_t = torch.from_numpy(mse.astype(np.float32))
    for s in range(S):
        out[s] = walk_monotone_curve(curve_t[s], list(grid), tau_vec,
                                     SIGMA_MIN, SIGMA_MAX).numpy()
    return out


def sigma_stats(sigma, tol=1e-6):
    v = sigma.flatten()
    at_min = float(np.mean(v <= SIGMA_MIN + tol))
    at_max = float(np.mean(v >= SIGMA_MAX - tol))
    inside = 1.0 - at_min - at_max
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return {"pct_in": inside * 100, "pct_at_min": at_min * 100, "pct_at_max": at_max * 100,
            "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}


def evaluate_tau(data, tau_per_subject, label, family, extra=None):
    """按逐被试 τ（绝对机制则各例同值）计算统计；返回逐被试行与汇总行。"""
    rows = []
    for d, tau in zip(data, tau_per_subject):
        st = sigma_stats(walk_sigma_matrix(d["mse"], d["grid"], tau))
        st.update({"subject": d["sid"], "tau": float(tau), "family": family, "label": label})
        if extra:
            st.update(extra)
        rows.append(st)
    pooled = {
        "family": family, "label": label, "n_subjects": len(rows),
        "tau_mean": float(np.mean([r["tau"] for r in rows])),
        "tau_min": float(np.min([r["tau"] for r in rows])),
        "tau_max": float(np.max([r["tau"] for r in rows])),
    }
    for k in ("pct_in", "pct_at_min", "pct_at_max", "median", "q25", "q75"):
        pooled[f"{k}_mean"] = float(np.mean([r[k] for r in rows]))
        pooled[f"{k}_sd"] = float(np.std([r[k] for r in rows]))
        pooled[f"{k}_min"] = float(np.min([r[k] for r in rows]))
        pooled[f"{k}_max"] = float(np.max([r[k] for r in rows]))
    if extra:
        pooled.update(extra)
    return rows, pooled


def _mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": C_INK2, "axes.labelcolor": C_INK, "text.color": C_INK,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
    return plt


def _tau_for(d, rec):
    """按推荐机制取该被试的 τ。"""
    if rec["family"] == "quantile":
        return compute_global_tau(
            torch.from_numpy(d["resp"][:, SIGMA_REFS.index(rec["sigma_ref"]), :].astype(np.float32)),
            rec["q"])
    return float(rec["tau_mean"])


def stage_scan(args):
    out_root = Path(args.out_root)
    curves_dir = out_root / "curves"
    stats_dir, plots_dir = out_root / "stats", out_root / "plots"
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(curves_dir, args.subjects)
    n_subj = len(data)
    assert n_subj > 0, "curves/ 为空，先跑 --stage curves"
    print(f"[scan] {n_subj} subjects loaded", flush=True)

    # ---- 校验：σ_ref=0.3 响应 vs run_02 raw_response_matrix（确定性复现检查） ----
    verif = []
    for d in data:
        p = RUN02_DIR / f"subject_{d['i']}.json"
        if not p.exists():
            continue
        old = np.array(json.load(open(p))["raw_response_matrix"], dtype=np.float64)
        rel = np.abs(d["resp"][:, 0, :] - old) / np.maximum(np.abs(old), 1e-12)
        verif.append({"subject": d["sid"], "max_rel_diff": float(rel.max())})
    print("[scan] 复现校验 vs run_02:", json.dumps(verif), flush=True)

    # ---- 结构诊断：floor/ceil/动态范围/单调性 ----
    diag = []
    for d in data:
        mse = d["mse"]
        floor, ceil = mse[:, 0, :], mse[:, -1, :]
        nonmono = float(np.mean(np.any(np.diff(mse, axis=1) < 0, axis=1)))
        diag.append({
            "subject": d["sid"],
            "floor_q05": float(np.quantile(floor, 0.05)),
            "floor_q50": float(np.quantile(floor, 0.5)),
            "floor_q95": float(np.quantile(floor, 0.95)),
            "ceil_q05": float(np.quantile(ceil, 0.05)),
            "ceil_q50": float(np.quantile(ceil, 0.5)),
            "ceil_q95": float(np.quantile(ceil, 0.95)),
            "dyn_ratio_q50": float(np.quantile(ceil / np.maximum(floor, 1e-12), 0.5)),
            "nonmono_pair_frac": nonmono,
        })

    # ---- τ 稠密梯子（向量化钳制比例；确定单全局 τ 可达上限） ----
    floors = np.concatenate([d["mse"][:, 0, :].flatten() for d in data])
    ceils = np.concatenate([d["mse"][:, -1, :].flatten() for d in data])
    lo = max(float(floors.min()) / 3.0, 1e-9)
    hi = float(ceils.max()) * 3.0
    ladder = np.exp(np.linspace(np.log(lo), np.log(hi), N_LADDER_DENSE))
    sweep = np.empty((len(ladder), 4))
    for k, tau in enumerate(ladder):
        cnt = np.zeros(3)
        for d in data:
            a, b, c = frac_split(d["mse"], tau)
            cnt += np.array([a.mean(), b.mean(), c.mean()])
        sweep[k] = [tau, *(cnt / n_subj)]
    tau_star_dense = float(sweep[np.argmax(sweep[:, 3]), 0])
    print(f"[scan] 稠密梯子上限: tau*={tau_star_dense:.4e} → interior="
          f"{sweep[:,3].max()*100:.1f}%", flush=True)

    # ---- 候选族评估（σ* 全部经 walk_monotone_curve） ----
    all_rows, all_pooled = [], []
    walk_ladder = list(np.exp(np.linspace(np.log(lo), np.log(hi), N_LADDER_WALK)))
    walk_ladder.append(tau_star_dense)                       # 稠密最优并入走查梯子
    for tau in walk_ladder:
        rows, pooled = evaluate_tau(data, [tau] * n_subj, f"abs={tau:.4e}", "ladder_abs")
        all_rows += rows
        all_pooled.append(pooled)
    for si, sref in enumerate(SIGMA_REFS):
        for q in Q_LADDER:
            taus = [compute_global_tau(torch.from_numpy(d["resp"][:, si, :].astype(np.float32)), q)
                    for d in data]
            rows, pooled = evaluate_tau(data, taus, f"q={q}@sref={sref}", "quantile",
                                        extra={"sigma_ref": sref, "q": q})
            all_rows += rows
            all_pooled.append(pooled)
    for c in SCALE_LADDER:
        taus = [c * compute_global_tau(torch.from_numpy(d["resp"][:, 0, :].astype(np.float32)), 0.5)
                for d in data]
        rows, pooled = evaluate_tau(data, taus, f"c={c}x(median@0.3)", "scale",
                                    extra={"scale_c": c, "sigma_ref": 0.3, "q": 0.5})
        all_rows += rows
        all_pooled.append(pooled)

    # run_02 口径行（对照锚点）
    taus_old = [compute_global_tau(torch.from_numpy(d["resp"][:, 0, :].astype(np.float32)), 0.75)
                for d in data]
    rows, pooled_old = evaluate_tau(data, taus_old, "run_02 (q=0.75@0.3)", "baseline_run02",
                                    extra={"sigma_ref": 0.3, "q": 0.75})
    all_rows += rows
    all_pooled.append(pooled_old)

    # ---- 推荐：interior 平台（±0.5pp 内统计平局）→ 与 07-04 正典 (q=0.5, σ_ref=0.3)
    # 偏差最小的分位数机制；退而求其次才用绝对 τ ----
    best_in = max(r["pct_in_mean"] for r in all_pooled)
    quant_plateau = [r for r in all_pooled
                     if r["family"] == "quantile" and r["pct_in_mean"] >= best_in - 0.5]
    if quant_plateau:
        rec = min(quant_plateau,
                  key=lambda r: (abs(r["q"] - 0.5) + abs(r["sigma_ref"] - 0.3),
                                 abs(r["pct_at_min_mean"] - r["pct_at_max_mean"])))
    else:
        rec = max((r for r in all_pooled if r["family"] == "ladder_abs"),
                  key=lambda r: r["pct_in_mean"])
    best_abs = max((r for r in all_pooled if r["family"] == "ladder_abs"),
                   key=lambda r: r["pct_in_mean"])
    best_quant = max((r for r in all_pooled if r["family"] == "quantile"),
                     key=lambda r: r["pct_in_mean"])
    print(f"[scan] 最优绝对 τ: {best_abs['label']} interior={best_abs['pct_in_mean']:.1f}%", flush=True)
    print(f"[scan] 最优分位数: {best_quant['label']} interior={best_quant['pct_in_mean']:.1f}%", flush=True)
    print(f"[scan] run_02 对照: interior={pooled_old['pct_in_mean']:.1f}%", flush=True)

    # ---- 落盘统计表 ----
    import csv
    keys = sorted({k for r in all_pooled for k in r})
    with open(stats_dir / "threshold_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_pooled)
    with open(stats_dir / "stats_per_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in all_rows for k in r}),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    per_subj_opt = []
    for d in data:
        s_lad, s_in = None, -1.0
        for tau in ladder:
            _, _, it = frac_split(d["mse"], tau)
            if it.mean() > s_in:
                s_in, s_lad = float(it.mean()), float(tau)
        per_subj_opt.append({"subject": d["sid"], "tau_star": s_lad, "pct_in_max": s_in * 100})
    with open(stats_dir / "optimum_and_diagnostics.json", "w") as fh:
        json.dump({"dense_ladder_best": {"tau": tau_star_dense,
                                         "pct_in_pooled": float(sweep[:, 3].max() * 100)},
                   "per_subject_optimum": per_subj_opt,
                   "structural_diag": diag,
                   "verification_vs_run02": verif}, fh, indent=2)

    # ---- 推荐配置 ----
    if rec["family"] == "quantile":
        rec_cfg = {"mechanism": "quantile", "sigma_ref": rec["sigma_ref"], "q": rec["q"],
                   "runner_repro_flags": f"--sigma-ref {rec['sigma_ref']} --q {rec['q']}"}
    else:
        rec_cfg = {"mechanism": "absolute", "tau": float(rec["tau_mean"]),
                   "runner_repro_note": "绝对 τ 暂无 runner CLI 旗标；用 --stage apply 离线产出"}
    rec_cfg.update({
        "pct_in_mean": rec["pct_in_mean"], "pct_in_sd": rec["pct_in_sd"],
        "pct_at_min_mean": rec["pct_at_min_mean"], "pct_at_max_mean": rec["pct_at_max_mean"],
        "sigma_star_median_mean": rec["median_mean"],
        "sigma_star_q25_mean": rec["q25_mean"], "sigma_star_q75_mean": rec["q75_mean"],
        "vs_run02": {"pct_in_mean": pooled_old["pct_in_mean"],
                     "pct_at_min_mean": pooled_old["pct_at_min_mean"],
                     "pct_at_max_mean": pooled_old["pct_at_max_mean"]},
        "search_space": {"sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX, "n_grid": N_GRID,
                         "sigma_refs": SIGMA_REFS, "q_ladder": Q_LADDER,
                         "scale_ladder": SCALE_LADDER, "n_abs_ladder_walk": N_LADDER_WALK},
    })
    with open(out_root / "recommended_config.json", "w") as fh:
        json.dump(rec_cfg, fh, indent=2)

    # ---- 绘图 ----
    _plots(data=data, sweep=sweep, rec=rec, pooled_old=pooled_old, plots_dir=plots_dir)
    print(f"[scan] done -> {stats_dir}, {plots_dir}, {out_root / 'recommended_config.json'}",
          flush=True)


def _plots(data, sweep, rec, pooled_old, plots_dir):
    plt = _mpl_setup()

    # 图 1：扰动强度-响应曲线（全对 + 中位带 + 阈值线 + 钳制边界）
    d0 = data[0]
    mse0 = d0["mse"]                                   # (S,G,C)
    grid = d0["grid"]
    S, G, C = mse0.shape
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    flat = mse0.reshape(S * C, G)
    for row in flat[::3]:                              # 稀疏化到 ~1300 条
        ax.plot(grid, row, color=C_INK2, alpha=0.05, lw=0.5)
    med = np.median(flat, axis=0)
    q25, q75 = np.quantile(flat, [0.25, 0.75], axis=0)
    ax.fill_between(grid, q25, q75, color=C_SERIES1, alpha=0.15, lw=0, label="IQR across pairs")
    ax.plot(grid, med, color=C_SERIES1, lw=2.0, label="median response")
    ax.axhline(pooled_old["tau_mean"], color=C_SERIES2, lw=1.6,
               label=f"run_02 τ (q=0.75@σ_ref=0.3) = {pooled_old['tau_mean']:.2e}")
    rec_tau = _tau_for(d0, rec)
    ax.axhline(rec_tau, color=C_SERIES3, lw=1.6, ls="--",
               label=f"recommended τ = {rec_tau:.2e}")
    ymin, ymax = ax.get_ylim()
    ax.axvline(SIGMA_MIN, color=C_CLAMP, lw=1.2, ls=":")
    ax.axvline(SIGMA_MAX, color=C_CLAMP, lw=1.2, ls=":")
    ax.text(SIGMA_MIN, ymax, " σ_min=0.01 (clamp)", color=C_CLAMP, va="top", fontsize=9)
    ax.text(SIGMA_MAX, ymax, "σ_max=0.48 (clamp) ", color=C_CLAMP, va="top", ha="right", fontsize=9)
    ax.axvline(0.3, color=C_INK2, lw=0.8, ls="--", alpha=0.6)
    ax.text(0.3, ymin, "σ_ref=0.3", color=C_INK2, fontsize=9, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("perturbation strength σ")
    ax.set_ylabel("per-target output MSE")
    ax.set_title(f"Perturbation strength–response curves ({d0['sid']}, {S}×{C} pairs, "
                 f"log grid [{SIGMA_MIN},{SIGMA_MAX}]×{G})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "response_curves.png", dpi=160)
    plt.close(fig)

    # 图 2：τ 扫描 —— 钳制/内点比例 vs τ
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(sweep[:, 0], sweep[:, 3] * 100, color=C_SERIES1, lw=2.0, label="interior σ*")
    ax.plot(sweep[:, 0], sweep[:, 1] * 100, color=C_SERIES2, lw=1.6, label="clamped @σ_min")
    ax.plot(sweep[:, 0], sweep[:, 2] * 100, color=C_SERIES3, lw=1.6, label="clamped @σ_max")
    tau_star = float(sweep[np.argmax(sweep[:, 3]), 0])
    ax.axvline(tau_star, color=C_INK, lw=1.2, ls="--")
    ax.annotate(f"τ* = {tau_star:.2e}\ninterior {sweep[:, 3].max() * 100:.1f}%",
                xy=(tau_star, sweep[:, 3].max() * 100), xytext=(8, -8),
                textcoords="offset points", fontsize=9, color=C_INK)
    ax.axvline(pooled_old["tau_mean"], color=C_SERIES2, lw=1.2, ls=":")
    ax.annotate(f"run_02 τ\n{pooled_old['pct_in_mean']:.1f}% interior",
                xy=(pooled_old["tau_mean"], 60), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color=C_SERIES2)
    ax.set_xscale("log")
    ax.set_xlabel("threshold τ (MSE units, log)")
    ax.set_ylabel("fraction of (a,b) pairs [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Threshold sweep: clamp / interior fractions (mean over subjects)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "tau_sweep.png", dpi=160)
    plt.close(fig)

    # 图 3：σ* ECDF —— 推荐 τ vs run_02 τ
    def ecdf(vals):
        v = np.sort(np.asarray(vals, dtype=np.float64).flatten())
        return v, np.arange(1, len(v) + 1) / len(v)

    sig_old, sig_rec = [], []
    for d in data:
        tau_o = compute_global_tau(torch.from_numpy(d["resp"][:, 0, :].astype(np.float32)), 0.75)
        sig_old.append(walk_sigma_matrix(d["mse"], d["grid"], tau_o))
        sig_rec.append(walk_sigma_matrix(d["mse"], d["grid"], _tau_for(d, rec)))
    v1, e1 = ecdf(np.concatenate(sig_old))
    v2, e2 = ecdf(np.concatenate(sig_rec))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.step(v1, e1, where="post", color=C_SERIES2, lw=1.8, label="run_02 τ (q=0.75@0.3)")
    ax.step(v2, e2, where="post", color=C_SERIES1, lw=1.8, label="recommended τ")
    ax.axvline(SIGMA_MIN, color=C_CLAMP, lw=1.2, ls=":")
    ax.axvline(SIGMA_MAX, color=C_CLAMP, lw=1.2, ls=":")
    ax.text(SIGMA_MIN * 1.1, 0.92, "σ_min", color=C_CLAMP, fontsize=9)
    ax.text(SIGMA_MAX * 0.92, 0.92, "σ_max", color=C_CLAMP, fontsize=9, ha="right")
    ax.axhline(0.5, color=C_GRID, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("σ* (log)")
    ax.set_ylabel("ECDF")
    ax.set_title("σ* distribution: recommended vs run_02 threshold (all subjects pooled)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "sigma_star_ecdf.png", dpi=160)
    plt.close(fig)

    # 图 4：逐被试 interior% 对比（哑铃图）
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ys = np.arange(len(data))
    for yi, d in zip(ys, data):
        tau_o = compute_global_tau(torch.from_numpy(d["resp"][:, 0, :].astype(np.float32)), 0.75)
        so = sigma_stats(walk_sigma_matrix(d["mse"], d["grid"], tau_o))
        sr = sigma_stats(walk_sigma_matrix(d["mse"], d["grid"], _tau_for(d, rec)))
        ax.plot([so["pct_in"], sr["pct_in"]], [yi, yi], color=C_GRID, lw=2.0, zorder=1)
        ax.scatter(so["pct_in"], yi, color=C_SERIES2, s=42, zorder=2)
        ax.scatter(sr["pct_in"], yi, color=C_SERIES1, s=42, zorder=3)
        ax.text(sr["pct_in"], yi + 0.22, f"{sr['pct_in']:.0f}%", fontsize=8,
                color=C_INK2, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([d["sid"] for d in data])
    ax.set_ylim(-0.6, float(ys.max()) + 0.8)
    ax.set_xlabel("interior σ* fraction [%]")
    ax.set_title("Per-subject interior fraction: run_02 τ (orange) → recommended τ (blue)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_subject_interior.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage C — 按推荐配置离线生成 subject JSON（前向确定性 ⇒ 离线 σ* ≡ 重跑）
# ---------------------------------------------------------------------------

def stage_apply(args):
    out_root = Path(args.out_root)
    curves_dir = out_root / "curves"
    rec_dir = out_root / "recommended"
    rec_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(out_root / "recommended_config.json"))
    data = load_all(curves_dir, args.subjects)
    log_2pie = math.log(2.0 * math.pi * math.e)
    for d in data:
        if cfg["mechanism"] == "quantile":
            tau = compute_global_tau(
                torch.from_numpy(
                    d["resp"][:, SIGMA_REFS.index(cfg["sigma_ref"]), :].astype(np.float32)),
                cfg["q"])
        else:
            tau = float(cfg["tau"])
        sigma_matrix = walk_sigma_matrix(d["mse"], d["grid"], tau)
        conn = -(np.log(sigma_matrix.astype(np.float64)) + log_2pie) + 1.0
        sign = np.sign(d["md"])
        st = sigma_stats(sigma_matrix)
        payload = {
            "subject_idx": d["i"], "subject_file": d["sid"],
            "method": ("global_tau threshold-search recommended "
                       f"({cfg['mechanism']}, sigma* preserved, offline from verified curves)"),
            "tau_global": float(tau),
            "sigma_matrix": sigma_matrix.tolist(),
            "connectivity_matrix": conn.tolist(),
            "connectivity_formula": "-(log sigma* + log(2*pi*e)) + 1",
            "sign_matrix": sign.tolist(),
            "signed_connectivity_matrix": (sign * conn).tolist(),
            "threshold_config": cfg,
            "hparams": {"sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX, "n_grid": N_GRID,
                        "sigma_refs_evaluated": SIGMA_REFS, "seed": SEED, "n_trials": N_TRIALS},
            "diagnostics": {"pct_at_sigma_min": st["pct_at_min"],
                            "pct_at_sigma_max": st["pct_at_max"],
                            "pct_interior": st["pct_in"],
                            "sigma_median": st["median"], "sigma_q25": st["q25"],
                            "sigma_q75": st["q75"]},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(rec_dir / f"subject_{d['i']}.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[apply] {d['sid']}: tau={tau:.4e} interior={st['pct_in']:.1f}% "
              f"@min={st['pct_at_min']:.1f}% @max={st['pct_at_max']:.1f}%", flush=True)


if __name__ == "__main__":
    stage = parse_args()
    {"curves": stage_curves, "scan": stage_scan, "apply": stage_apply}[stage.stage](stage)
