#!/usr/bin/env python3
"""run_search.py — perturb-search-v1 入口（design §6 两阶段；B1 smoke / B2 phase1 / B3 phase2）。

用法（implement.md）：
  cd models/perturb-search-v1
  python3 run_search.py smoke                          # B1：1 窗阶段一批（559 行）
  python3 run_search.py phase1                         # B2：24 窗 × 双探针 {G, N} → 选 M*
  python3 run_search.py phase2                         # B3：24 窗 × 6 范式 × 三度量集（config.PHASE2_METRICS）
  python3 run_search.py phase2 --metric M-RMS M-time   # 覆盖度量集（须属 config.METRICS）
  nohup python3 run_search.py phase1 > ../../outputs/perturb-search-f0/logs/phase1.log 2>&1 &

流程：BLAS 护栏 → θ* 冻结装载（config_hash 校验 + EMA）→ 逐窗固定枚举（win_idx =
受试者序 × 8 + 例内窗序）→ 变体批（559/1737 行 = 1 + 62 源 × Σ范式 r 格点数，一次
sample()/窗，显式 eps/eta ⇒ 跨 r CRN 逐位配对）→ 6 度量 + null → scorecard（Gate
A/B/C @ r0 格点切片 + Gate D 全网格 |Δ| 单调性 + 选择规则）→ npz + csv + summary.md 落盘。

phase2 硬断言（design §6）：同窗 eps/eta 跨阶段逐位相同（sha256 逐窗比对）；baseline
输出紧容差 allclose rtol=2e-6 对照登记（不作逐位门；逐位相等如实测得则照登）。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy/torch 导入之前；----
# ---- 有效 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）            ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import csv       # noqa: E402
import hashlib   # noqa: E402
import json      # noqa: E402
import subprocess  # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import candidates as CD   # noqa: E402  本目录（search_config 唯一装载 + 算子 + crn_noise）
import loader as LD       # noqa: E402  触发 v1 sample/data/model/gpr_prior 只读装配
import metrics as MT      # noqa: E402
import scorecard as SG    # noqa: E402

# 可解释性/注意注记（design §3/§4「一句话理由」转录，随 scorecard 照登）
METRIC_NOTES = {
    "M-RMS": "现行 D 同构基准（时域幅度）",
    "M-PSD": "谱域完整描述；对波形相位不敏感",
    "M-band": "EEG 频带可解释性强；δ 带 1–4 Hz 仅 ~1–2 bin 方差大（已知照登）",
    "M-env": "振幅耦合载体（连接文献常用）；128 样本 Hilbert 边缘效应（pert/base/null 同算子，部分抵消）",
    "M-FC": "网络级/同步响应；依赖全通道（网络态度量非纯靶响应）",
    "M-time": "判响应起点与时程；瞬态探针天然读出",
}
PARADIGM_NOTES = {
    "G": "现行 A 案参照；r≥0.32 外推警示（r0=0.25 界内）",
    "N": "perturb-and-record 标准形态；直接注入方差",
    "O": "最简加性（DC 平移）；数据 1–40 Hz 带限下 DC 属外推输入（解释警示照登）",
    "P": "瞬态刺激 → 时间形态可辨；剂量高因持续时间短",
    "L": "损毁式（向自身均值收缩 50%）；与增益互补方向，数值安全不外推",
    "B": "频率选择性（α 带分量增益）；窗 RMS 足迹最小（~0.15–0.27σ，地板化风险照登）",
}


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256_t(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def _rows_to_metric_matrix(D_rows: np.ndarray, n_rt: int) -> np.ndarray:
    """compute_metrics 的行（行序 v-1 = a·N_RT + j）→ 格点主序矩阵。
    (V-1, C) → (N_RT, A, B)；(V-1, C, nb) → (N_RT, A, B, nb)（行=源、列=靶）。"""
    SC = CD.load_config()
    M = D_rows.reshape(SC.N_SRC, n_rt, *D_rows.shape[1:])
    axes = (1, 0) + tuple(range(2, M.ndim))
    return M.transpose(axes).copy()


def _fmt(v, spec: str = ".4g") -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else format(v, spec)
    return str(v)


def _write_csv(path: Path, rows: list, fields: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _write_meta(out_dir: Path, cmd: str, extra: dict) -> None:
    SC = CD.load_config()
    cfg_dump = {k: v for k, v in vars(SC).items() if k.isupper()}
    meta = {"command": cmd, "git_head": _git_head(), "timestamp": datetime.now().isoformat(),
            "config": cfg_dump, **extra}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 主循环（smoke = phase1 机制 · 1 窗）
# ---------------------------------------------------------------------------

def _run_stage(stage: str, metric_arg: list | None) -> None:
    SC = CD.load_config()
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    t_wall0 = time.perf_counter()
    is_phase2 = stage == "phase2"
    is_smoke = stage == "smoke"
    out_root = Path(SC.OUTPUTS_ROOT)
    if is_smoke:
        out_dir = out_root / SC.SMOKE_DIR
        paradigms, metrics = SC.PHASE1_PROBES, tuple(SC.PHASE1_METRICS)
    elif is_phase2:
        if metric_arg:
            assert all(a in SC.METRICS for a in metric_arg), \
                f"--metric 须属 {SC.METRICS}，收到 {metric_arg!r}"
        metrics = tuple(metric_arg) if metric_arg else tuple(SC.PHASE2_METRICS)
        out_dir = out_root / SC.PHASE2_DIR
        paradigms = tuple(SC.PHASE2_PARADIGMS)
    else:
        out_dir = out_root / SC.PHASE1_DIR
        paradigms, metrics = SC.PHASE1_PROBES, tuple(SC.PHASE1_METRICS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_root / SC.LOG_DIR).mkdir(parents=True, exist_ok=True)

    # ---- phase2 跨阶段对照数据（design §6：硬断言 + 紧容差对照）----
    ph1 = None
    if is_phase2:
        p1 = np.load(out_root / SC.PHASE1_DIR / "per_edge_D.npz", allow_pickle=False)
        ph1 = {"eps": p1["eps_sha256"].astype(str), "eta": p1["eta_sha256"].astype(str),
               "base": p1["baseline_outputs"]}

    # ---- 装载 ----
    print(f"[{stage}] 装载 θ*（EMA + config_hash 校验）…", flush=True)
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    assert sampler.gen_lowpass_hz is None, "gen_lowpass_hz 须 None（fmt 口径；fs=500 brickwall 坑）"
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    _lo, _lds = LD.load_test_loader()
    wins = LD.select_windows(_lds)
    if is_smoke:
        wins = wins[:1]
    W = len(wins)
    specs = CD.variant_specs(paradigms)
    n_rt = len(specs)
    n_rows = 1 + SC.N_SRC * n_rt
    # r0 格点行（Gate A/B/C 切片）与逐范式格点段（Gate D 全网格；段内连续 = specs 范式主序）
    j0 = {pr: specs.index((p_i, pr, SC.R0[pr])) for p_i, pr in enumerate(paradigms)}
    rseg = {pr: (sum(len(SC.R_GRID[q]) for q in paradigms[:p_i]), len(SC.R_GRID[pr]))
            for p_i, pr in enumerate(paradigms)}
    K, C, F = SC.K, SC.N_CHANNELS, cfg.future_len
    print(f"[{stage}] 窗集 {W} × 批 {n_rows} 行（62 源 × {n_rt} 格点/源）× K={K}；"
          f"范式 {paradigms}；度量 {metrics}", flush=True)

    # ---- 结果容器 ----
    D_store = {m: np.zeros((n_rt, W, SC.N_SRC, SC.N_CHANNELS), np.float32) for m in metrics}
    null_store = {m: np.zeros((W, SC.N_CHANNELS), np.float32) for m in metrics}
    extra_store: dict[str, np.ndarray] = {}
    if "M-band" in metrics:
        for pr in paradigms:
            extra_store[f"band_delta_{pr}"] = np.zeros(
                (W, len(SC.BANDS), SC.N_SRC, SC.N_CHANNELS), np.float32)
    if "M-time" in metrics:
        for pr in paradigms:
            extra_store[f"mtime_full_{pr}"] = np.zeros((W, SC.N_SRC, SC.N_CHANNELS), np.float32)
            extra_store[f"mtime_peak_{pr}"] = np.zeros((W, SC.N_SRC, SC.N_CHANNELS), np.float32)
    baseline_out = np.zeros((W, K, C, F), np.float32)
    eps_hash = np.zeros(W, dtype="<U64")
    eta_hash = np.zeros(W, dtype="<U64")
    op_seconds = {pr: 0.0 for pr in paradigms}
    metric_seconds = {m: 0.0 for m in metrics}
    sample_seconds = 0.0
    xphase_report = []          # phase2 跨阶段对照逐窗登记

    # ---- 逐窗主循环：每窗一次 sample() ----
    for w, (win_idx, ds_idx, sid, wsub) in enumerate(wins):
        xp, _xf = _lds[ds_idx]
        xp = xp.to("cuda", non_blocking=True)
        eps, eta = CD.crn_noise(win_idx, K, C, F, "cuda")
        eps_hash[w], eta_hash[w] = _sha256_t(eps), _sha256_t(eta)
        if is_phase2:                                   # 硬断言：同窗 eps/eta 跨阶段逐位相同
            assert eps_hash[w] == ph1["eps"][w] and eta_hash[w] == ph1["eta"][w], \
                f"窗 {w} (win_idx={win_idx}) eps/eta sha256 与 phase1 不符（CRN 跨阶段契约破坏）"
        t0 = time.perf_counter()
        batch = CD.build_variant_batch(xp, paradigms, timing=op_seconds)
        assert batch.shape == (n_rows, C, cfg.past_len)
        t1 = time.perf_counter()
        with ema.swap(model):
            out = sampler.sample(batch, K, eps=eps, eta=eta)
        t_sample = time.perf_counter() - t1
        sample_seconds += t_sample
        assert out.shape == (n_rows, K, C, F), f"输出形状 {tuple(out.shape)} != ({n_rows},{K},{C},{F})"
        assert torch.isfinite(out).all(), "采样输出含非有限值"
        baseline_out[w] = out[0].float().cpu().numpy()
        if is_phase2:
            b2 = baseline_out[w].astype(np.float64)
            b1 = ph1["base"][w].astype(np.float64)
            ok = np.allclose(b2, b1, rtol=SC.CROSS_PHASE_RTOL)   # 紧容差（非逐位门）
            bitwise = np.array_equal(baseline_out[w], ph1["base"][w])
            denom = np.abs(b1)
            rel = np.abs(b2 - b1) / np.where(denom > 1e-6, denom, np.nan)
            xphase_report.append({"w": w, "allclose": bool(ok), "bitwise": bool(bitwise),
                                  "max_rel": float(np.nanmax(rel))})
            assert ok, f"窗 {w} baseline 跨阶段 allclose(rtol={SC.CROSS_PHASE_RTOL}) 失败"
        samples = out.cpu().numpy().astype(np.float64)
        for m in metrics:                               # 逐度量计时（PSD 家族重复计算照登成本）
            t2 = time.perf_counter()
            res = MT.compute_metrics(samples, [m])
            metric_seconds[m] += time.perf_counter() - t2
            D_store[m][:, w] = _rows_to_metric_matrix(res[m]["D"], n_rt)
            null_store[m][w] = res[m]["null"]
            if m == "M-band":                           # 登记项只取 r0 切片（npz 体积控制）
                bd = _rows_to_metric_matrix(res[m]["band_delta"], n_rt)   # (n_rt, A, B, 5)
                for pr in paradigms:
                    extra_store[f"band_delta_{pr}"][w] = bd[j0[pr]].transpose(2, 0, 1)  # (5, A, B)
            if m == "M-time":
                for key in ("mtime_full", "mtime_peak"):
                    mm = _rows_to_metric_matrix(res[m][key], n_rt)
                    for pr in paradigms:
                        extra_store[f"{key}_{pr}"][w] = mm[j0[pr]]
        el = time.perf_counter() - t_wall0
        eta_min = el / (w + 1) * (W - w - 1) / 60
        print(f"[{stage}] 窗 {w + 1}/{W}（sid={sid} win_idx={win_idx} 例内 {wsub}）"
              f" | 累计 {el / 60:.1f} min | ETA {eta_min:.1f} min | sample {t_sample:.1f}s",
              flush=True)

    # ---- Gate D（全网格 |Δ| 单调性；design §5 增补）----
    gd = {}
    for m in metrics:
        for pr in paradigms:
            s, n_r = rseg[pr]
            gd[(pr, m)] = SG.gate_d(D_store[m][s:s + n_r])   # (R, W, A, B) 段

    # ---- 落盘 npz ----
    npz_path = out_dir / "per_edge_D.npz"
    arrays = {}
    for m in metrics:
        for pr in paradigms:
            s, n_r = rseg[pr]
            for i, r in enumerate(SC.R_GRID[pr]):        # 逐格点 D 矩阵（Gate D 复算数据源）
                arrays[f"D_{pr}_r{r:g}_{m}"] = D_store[m][s + i]
            arrays[f"gateD_curve_{pr}_{m}"] = gd[(pr, m)]["curve"]
        arrays[f"null_{m}"] = null_store[m]
    arrays.update(extra_store)
    if not is_phase2:                                   # phase1 存 baseline 供 phase2 对照
        arrays["baseline_outputs"] = baseline_out
    for pr in paradigms:
        arrays[f"r_grid_{pr}"] = np.array(SC.R_GRID[pr])
    arrays.update({
        "eps_sha256": eps_hash, "eta_sha256": eta_hash,
        "subject_ids": np.array([sid for _w, _ds, sid, _ws in wins]),
        "win_idx": np.array([wi for wi, _d, _s, _w in wins]),
        "ds_idx": np.array([d for _w, d, _s, _w in wins]),
        "window_in_subject": np.array([ws for _w, _d, _s, ws in wins]),
        "paradigms": np.array(paradigms), "metrics": np.array(metrics),
        "K": np.int64(K), "seed": np.int64(SC.SEED), "batch_rows": np.int64(n_rows),
        "n_grid_points": np.int64(n_rt), "r0_rows": np.array([j0[pr] for pr in paradigms]),
    })
    np.savez(npz_path, **arrays)

    # ---- scorecard（A/B/C @ r0 切片；D 全网格）----
    wall = time.perf_counter() - t_wall0
    rows12: list[dict] = []          # CSV 行（phase1: 度量×探针 12 行；phase2/smoke: 范式行）
    cand_rows: list[dict] = []       # 选择用候选级行
    if not is_phase2:
        for m in metrics:
            ev = SG.evaluate_candidate([D_store[m][j0[pr]] for pr in paradigms],
                                       null_store[m], coords)
            gbest = min((gd[(pr, m)] for pr in paradigms),            # Gate D 逐探针判（or 口径），
                        key=lambda g: (not g["pass"], len(g["viol_steps"])))   # 诊断取较优探针
            metric_pass = (ev["gateA"][0]["pass"] or ev["gateA"][1]["pass"]) and \
                ev["gateB"]["pass"] and ev["gateC"]["pass"] and gbest["pass"]
            cr = {"metric": m, "gateA_median_z": [g["median_z"] for g in ev["gateA"]],
                  "gateB_ratio": ev["gateB"]["ratio"], "gateB_pass": ev["gateB"]["pass"],
                  "gateC_rho": ev["gateC"]["rho"], "gateC_pass": ev["gateC"]["pass"],
                  "gateD_pass": gbest["pass"],
                  "gateD_viol_steps": len(gbest["viol_steps"]), "gateD_n_steps": gbest["n_steps"],
                  "gateD_edge_mono_frac": gbest["edge_mono_frac"],
                  "all_gates_pass": metric_pass,
                  "cost_seconds": metric_seconds[m], "note": METRIC_NOTES[m]}
            cand_rows.append(cr)
            for p_i, pr in enumerate(paradigms):
                g = gd[(pr, m)]
                rows12.append({
                    "metric": m, "probe": pr, "r0": SC.R0[pr],
                    "gateA_median_z": ev["gateA"][p_i]["median_z"],
                    "gateA_frac_z_ge2": ev["gateA"][p_i]["frac_z_ge2"],
                    "gateA_pass": ev["gateA"][p_i]["pass"],
                    "gateB_ratio": ev["gateB"]["ratio"],
                    "gateB_spearman_dist": ev["gateB"]["spearman_dist"],
                    "gateB_pass": ev["gateB"]["pass"],
                    "gateC_rho": ev["gateC"]["rho"], "gateC_pass": ev["gateC"]["pass"],
                    "gateD_pass": g["pass"],
                    "gateD_viol_steps": len(g["viol_steps"]), "gateD_n_steps": g["n_steps"],
                    "gateD_edge_mono_frac": g["edge_mono_frac"],
                    "all_gates_pass": metric_pass,
                    "cost_seconds": metric_seconds[m],
                    "interp_rank": SC.METRIC_INTERP_ORDER.index(m) + 1, "note": METRIC_NOTES[m]})
        key_name, interp_order = "metric", SC.METRIC_INTERP_ORDER
    else:
        for m in metrics:
            for pr in paradigms:
                ev = SG.evaluate_candidate([D_store[m][j0[pr]]], null_store[m], coords)
                g = gd[(pr, m)]
                allp = ev["gateA"][0]["pass"] and ev["gateB"]["pass"] and ev["gateC"]["pass"] \
                    and g["pass"]
                row = {"metric": m, "paradigm": pr, "r0": SC.R0[pr],
                       "gateA_median_z": ev["gateA"][0]["median_z"],
                       "gateA_frac_z_ge2": ev["gateA"][0]["frac_z_ge2"],
                       "gateA_pass": ev["gateA"][0]["pass"],
                       "gateB_ratio": ev["gateB"]["ratio"],
                       "gateB_spearman_dist": ev["gateB"]["spearman_dist"],
                       "gateB_pass": ev["gateB"]["pass"],
                       "gateC_rho": ev["gateC"]["rho"], "gateC_pass": ev["gateC"]["pass"],
                       "gateD_pass": g["pass"],
                       "gateD_viol_steps": len(g["viol_steps"]), "gateD_n_steps": g["n_steps"],
                       "gateD_edge_mono_frac": g["edge_mono_frac"],
                       "all_gates_pass": allp, "cost_seconds": op_seconds[pr],
                       "interp_rank": SC.PARADIGM_INTERP_ORDER.index(pr) + 1,
                       "note": PARADIGM_NOTES[pr]}
                rows12.append(row)
                cand_rows.append(dict(row))
        key_name, interp_order = "paradigm", SC.PARADIGM_INTERP_ORDER

    fields = (["metric", "probe"] if not is_phase2 else ["metric", "paradigm"]) + \
             ["r0", "gateA_median_z", "gateA_frac_z_ge2", "gateA_pass", "gateB_ratio",
              "gateB_spearman_dist", "gateB_pass", "gateC_rho", "gateC_pass",
              "gateD_pass", "gateD_viol_steps", "gateD_n_steps", "gateD_edge_mono_frac",
              "all_gates_pass", "cost_seconds", "interp_rank", "note"]
    _write_csv(out_dir / "scorecard.csv", rows12, fields)

    winner = None
    winners_by_metric: list = []      # phase2：[(metric, winner|None, cand_rows), ...] 逐度量独立判定
    if not is_smoke:
        if not is_phase2:
            winner = SG.select_winner(cand_rows, interp_order, key_name)
        else:
            for m in metrics:
                sub = [r for r in cand_rows if r["metric"] == m]
                winners_by_metric.append((m, SG.select_winner(sub, interp_order, key_name), sub))
    # ---- summary.md ----
    lines = [f"# {'smoke' if is_smoke else stage} — perturb-search-v1（v1-fmt θ* {SC.EXPECTED_CONFIG_HASH}）",
             "",
             f"- 窗集：{W} 窗（受试者 {sorted(set(int(s) for s in arrays['subject_ids']))} × "
             f"前 {SC.N_WINDOWS_PER_SUBJECT} 窗，win_idx 固定枚举）；K={K}；seed={SC.SEED}；"
             f"批 {n_rows} 行（62 源 × {n_rt} 格点/源 = 范式 {'/'.join(paradigms)} × r 网格）"
             f"一次 sample()/窗（跨 r CRN 逐位配对 = 「仅改变 r」）。",
             f"- r 网格（×r0 倍率梯 {SC.R_GRID_MULT}，钳制 {SC.R_CLAMP}）："
             + "；".join(f"{pr} {['%g' % r for r in SC.R_GRID[pr]]}" for pr in paradigms) + "。",
             f"- 耗时：wall {wall / 60:.1f} min（sample 累计 {sample_seconds:.1f}s"
             + (f"；度量 {', '.join(f'{m} {metric_seconds[m]:.1f}s' for m in metrics)}" if not is_phase2
                else f"；度量 {', '.join(f'{m} {metric_seconds[m]:.1f}s' for m in metrics)}"
                     f"；算子施加 {', '.join(f'{pr} {op_seconds[pr]:.1f}s' for pr in paradigms)}") + "）。",
             f"- ckpt：{SC.CKPT_PATH}（epoch {ck['epoch']}，config_hash {ck['config_hash']}）；"
             f"gen_lowpass_hz=None 断言过。", ""]
    if is_phase2:
        n_bit = sum(r["bitwise"] for r in xphase_report)
        worst = max(r["max_rel"] for r in xphase_report)
        lines += [f"- 跨阶段硬断言：24/24 窗 eps/eta sha256 与 phase1 逐窗相同（逐位）；"
                  f"baseline 输出 allclose(rtol={SC.CROSS_PHASE_RTOL}) 24/24 过，"
                  f"逐位相等 {n_bit}/24 窗（如实照登），逐窗 max_rel 最大 {worst:.3g}。", ""]
    lines += ["## scorecard（Gate A z 中位 / Gate B 自跨比 / Gate C split-half ρ / Gate D 单调；"
              f"门 = A≥{SC.GATE_A_Z}, B≥{SC.GATE_B_RATIO}, C≥{SC.GATE_C_RHO}, "
              "D=聚合曲线 100% 步不减）", "",
              "| " + ("候选 | 探针 | " if not is_phase2 else "度量 | 候选 | ") + "A_z | A 过 | B 比 | "
              "B ρ_dist | C ρ | D 过 | D 违反步 | D 边单调占比 | 全过 | 成本 s | 注记 |",
              "|---|" + "---|" * (12 if not is_phase2 else 12)]
    for r in rows12:
        gal = r["gateA_median_z"]
        prefix = f"{r[key_name]} | {r['probe']} | " if not is_phase2 \
            else f"{r['metric']} | {r[key_name]} | "
        lines.append(prefix +
                     f"{_fmt(gal)} | {r['gateA_pass']} | {_fmt(r['gateB_ratio'])} | "
                     f"{_fmt(r['gateB_spearman_dist'])} | {_fmt(r['gateC_rho'])} | "
                     f"{r['gateD_pass']} | {r['gateD_viol_steps']}/{r['gateD_n_steps']} | "
                     f"{_fmt(r['gateD_edge_mono_frac'])} | "
                     f"{r['all_gates_pass']} | {r['cost_seconds']:.1f} | {r['note']} |")
    lines += [""]
    viol_lines = []
    for m in metrics:
        for pr in paradigms:
            g = gd[(pr, m)]
            grid = SC.R_GRID[pr]
            for k in g["viol_steps"]:
                viol_lines.append(
                    f"- {pr}×{m}：步 {k}→{k + 1}（r {grid[k]:g}→{grid[k + 1]:g}）曲线 "
                    f"{g['curve'][k]:.4g}→{g['curve'][k + 1]:.4g}"
                    f"（相对降幅 {(g['curve'][k + 1] / g['curve'][k] - 1) * 100:+.1f}%）")
    if viol_lines:
        lines += ["## Gate D 违反明细（违反单调性的情形；聚合曲线口径）", ""] + viol_lines + [""]
    if is_smoke:
        lines += ["- smoke 只验证机械正确性（W=1 时 B/C 门 NaN 不判；Gate A 为单窗 std=NaN 不判；"
                  "Gate D 单窗机械可算但不作判定）。"]
    elif not is_phase2:
        if winner is not None:
            passers = [r for r in cand_rows if r["all_gates_pass"]]
            order = {k: i for i, k in enumerate(interp_order)}
            passers.sort(key=lambda r: (-r["gateB_ratio"], -r["gateC_rho"], order[r[key_name]],
                                        r["cost_seconds"], r[key_name]))
            lines += [f"## 结论：M\\* = {winner[key_name]}",
                      "",
                      f"- 过线候选（四门全过）：{[r[key_name] for r in passers]}；按选择规则（四门全过为"
                      f"入场券，Gate B 主排序 → Gate C → 可解释性 → 成本）选出 **M\\* = "
                      f"{winner[key_name]}**（Gate B 比 {winner['gateB_ratio']:.3f}，"
                      f"Gate C ρ {winner['gateC_rho']:.3f}，Gate D 违反步 "
                      f"{winner.get('gateD_viol_steps')}/{winner.get('gateD_n_steps')}）。"]
        else:
            lines += ["## 结论：如实阴性（全候选未过线）", "", SG.negative_diagnosis(cand_rows, key_name)]
    else:
        lines += ["## 结论：逐度量 P\\*（联合 {P×M} 终选以矩阵呈报，留用户拍板）", ""]
        for m, w, sub in winners_by_metric:
            if w is not None:
                passers = [r for r in sub if r["all_gates_pass"]]
                order = {k: i for i, k in enumerate(interp_order)}
                passers.sort(key=lambda r: (-r["gateB_ratio"], -r["gateC_rho"], order[r[key_name]],
                                            r["cost_seconds"], r[key_name]))
                lines += [f"- **{m}：P\\* = {w[key_name]}**（过线范式 {[r[key_name] for r in passers]}；"
                          f"Gate B 比 {w['gateB_ratio']:.3f}，Gate C ρ {w['gateC_rho']:.3f}，"
                          f"Gate D 违反步 {w['gateD_viol_steps']}/{w['gateD_n_steps']}）"]
            else:
                lines += [f"- **{m}：如实阴性**", "", SG.negative_diagnosis(sub, key_name), ""]
        lines += ["- 结论限定「各自 r0 下」（A/B/C @ r0 单点 ⇒ P\\* 实为「范式×r0」联合选择，"
                  "不得表述为全域最优范式；Gate D 已在各自 r 网格验单调，不改变该限定）。",
                  f"- 度量集 {tuple(metrics)} = 用户 2026-10-18 拍板（阶段一阴性无单 M\\* 的显式偏离，"
                  "design §6 登记）。"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_meta(out_dir, " ".join(["python3 run_search.py", stage] +
                                  (["--metric"] + list(metric_arg) if metric_arg else [])), {
        "wall_seconds": wall, "sample_seconds": sample_seconds,
        "metric_seconds": metric_seconds, "op_seconds": op_seconds,
        "ckpt_epoch": int(ck["epoch"]), "ckpt_config_hash": ck["config_hash"],
        "euler_steps": cfg.euler_steps, "sigma_min": cfg.sigma_min,
        "gen_lowpass_hz": getattr(cfg, "gen_lowpass_hz", 40.0),
        "xphase": xphase_report if is_phase2 else None,
    })
    print(f"[{stage}] 完成：wall {wall / 60:.1f} min（sample {sample_seconds:.1f}s）→ {out_dir}",
          flush=True)
    if winner is not None:
        print(f"[{stage}] 选出 M\\* = {winner[key_name]}", flush=True)
    elif is_phase2:
        for m, w, sub in winners_by_metric:
            print(f"[{stage}] {m}: P\\* = {w[key_name] if w is not None else '阴性'}", flush=True)
    elif not is_smoke:
        print(f"[{stage}] 如实阴性：\n{SG.negative_diagnosis(cand_rows, key_name)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["smoke", "phase1", "phase2"])
    p.add_argument("--metric", type=str, nargs="+", default=None,
                   help="phase2 度量集覆盖（缺省 = config.PHASE2_METRICS，须各属 METRICS）")
    args = p.parse_args()
    _run_stage(args.stage, args.metric)


if __name__ == "__main__":
    main()
