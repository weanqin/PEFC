#!/usr/bin/env python3
"""diag_hf.py — 高频（40 Hz+）功率偏高根因诊断（Run 012；只读诊断，不改任何口径）。

分解生成管线上各环节的功率谱，量化「真实侧被 v4 滤波压零、生成侧仍高」的来源：
  real        真实 future（z-score 同口径）
  mu          GPR 后验均值 μ = A·y^p（先验确定性部分，线性算子）
  x0_iso      条件先验样本（iso=1e-2，训练/推理同配置）
  x0_noiso    条件先验样本（iso=0——隔离白噪折叠贡献；仅诊断构造）
  ep{8,16,32,64}  ODE 终点 x̂(1)（EMA 权重，不同 Euler 步数）
  gen_v1/gen_v2   两 run 的 gc_input.npz 生成侧（K=1，全量 13210 窗）

频段 {40–100, 100–250, >40} 功率占比 + 低通（30/40 Hz 砖墙）后 MAE 变化。
产物：outputs/gpfm-mamba-f0/diag-hf/{diag_hf.json, diag_hf.npz}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # 仓库根

import config as C  # noqa: E402
from gpr_prior import GPRPrior  # noqa: E402
from sample import (EulerSampler, build_eval_prior, iterate_test_batches,  # noqa: E402
                    load_eval_model)

FREQ_RES = 500.0 / 128.0   # 周期图分辨率 3.90625 Hz
DIAG_SEED = 2027


def periodogram_avg(x: np.ndarray) -> np.ndarray:
    """x (N,C,128) → 全元素平均单边功率谱（Hann 周期图；内部可比口径）。"""
    seg = np.asarray(x, dtype=np.float64)
    w = np.hanning(seg.shape[-1])
    p = np.abs(np.fft.rfft(seg * w, axis=-1)) ** 2
    return p.mean(axis=(0, 1))


def band_stats(freqs: np.ndarray, psd: np.ndarray) -> dict:
    tot = psd[1:].sum()          # 去 DC
    out = {"var_total": float(tot)}
    for name, lo, hi in (("40_100", 40, 100), ("100_250", 100, 250), ("gt40", 40, 250)):
        m = (freqs >= lo) & (freqs < hi)
        out[f"frac_{name}"] = float(psd[m].sum() / tot)
        out[f"psd_{name}"] = float(psd[m].mean())
    # 真实谱拐点：功率降到峰值 1% 处的频率
    peak = psd[1:].max()
    idx = np.where(psd[1:] < 0.01 * peak)[0]
    out["real_corner_est_hz"] = float(freqs[1:][idx[0]]) if len(idx) else None
    return out


def lowpass_mae(gen: np.ndarray, real: np.ndarray, cut: float) -> float:
    """128 点段 rFFT 砖墙低通（>cut Hz 置零），返回滤波后与 real 的 MAE。"""
    freqs = np.fft.rfftfreq(gen.shape[-1], d=1.0 / 500.0)
    spec = np.fft.rfft(gen, axis=-1)
    spec[..., freqs > cut] = 0
    g = np.fft.irfft(spec, n=gen.shape[-1], axis=-1)
    return float(np.abs(g - real).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default="main-01")
    ap.add_argument("--n-windows", type=int, default=512)
    ap.add_argument("--n-sub-steps", type=int, default=256)  # 步数敏感性子集窗数
    args = ap.parse_args()

    cfg = C.build_config()
    device = "cuda"
    assert torch.cuda.is_available()
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    run_dir = Path(cfg.outputs_root) / args.run_id
    out_dir = Path(cfg.outputs_root) / "diag-hf"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ema, ck = load_eval_model(
        str(run_dir / "checkpoints" / f"{cfg.variant_name}-best.pt"), cfg, device)
    prior = build_eval_prior(cfg, device)
    prior_noiso = GPRPrior(
        np.load(f"{cfg.data_root}/chan_coords_62.npy"),
        past_len=cfg.past_len, future_len=cfg.future_len,
        kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
        iso=0.0, pp_jitter=cfg.pp_jitter,
        sigma_s=cfg.sigma_s, jitter_s=cfg.jitter_s,
    ).to(device)
    loader, test_ds = iterate_test_batches(cfg, run_dir, batch_size=64)

    elems = {k: [] for k in ("real", "mu", "x0_iso", "x0_noiso", "ep16")}
    g0 = torch.Generator(device=device); g0.manual_seed(DIAG_SEED)
    n_col = 0
    t0 = time.time()
    with ema.swap(model), torch.no_grad():
        sampler = EulerSampler(model, prior, euler_steps=cfg.euler_steps,
                               sigma_min=cfg.sigma_min)
        for xp, x1 in loader:
            xp = xp.to(device); x1 = x1.to(device)
            mu = prior.posterior_mean(xp)
            x0a = prior.sample(xp, generator=g0)
            x0b = prior_noiso.sample(xp, generator=g0)
            ep = sampler.sample(xp, 1, generator=g0)[:, 0]
            elems["real"].append(x1.cpu().numpy())
            elems["mu"].append(mu.cpu().numpy())
            elems["x0_iso"].append(x0a.cpu().numpy())
            elems["x0_noiso"].append(x0b.cpu().numpy())
            elems["ep16"].append(ep.cpu().numpy())
            n_col += xp.shape[0]
            if n_col >= args.n_windows:
                break
    arrays = {k: np.concatenate(v)[: args.n_windows] for k, v in elems.items()}
    print(f"[diag] 采集 {n_col} 窗 × 5 元素 | {time.time()-t0:.1f}s", flush=True)

    # Euler 步数敏感性（子集，多步数变体）
    sub = {k: v[: args.n_sub_steps] for k, v in arrays.items()}
    xp_sub = None  # 重新取前 n_sub 窗 past
    xs = []
    ys = []
    n_sub = 0
    for xp, x1 in loader:
        xs.append(xp); ys.append(x1)
        n_sub += xp.shape[0]
        if n_sub >= args.n_sub_steps:
            break
    xp_sub = torch.cat(xs)[: args.n_sub_steps].to(device)
    g1 = torch.Generator(device=device); g1.manual_seed(DIAG_SEED)
    steps_out = {}
    with ema.swap(model), torch.no_grad():
        for nst in (8, 32, 64):
            s = EulerSampler(model, prior, euler_steps=nst, sigma_min=cfg.sigma_min)
            g1.manual_seed(DIAG_SEED)
            steps_out[f"ep{nst}"] = s.sample(xp_sub, 1, generator=g1)[:, 0].cpu().numpy()
    print(f"[diag] 步数敏感性 {sorted(steps_out)} 完成", flush=True)

    freqs = np.fft.rfftfreq(128, d=1.0 / 500.0)
    psd_all = {}
    stats = {}
    for k, v in arrays.items():
        psd_all[k] = periodogram_avg(v)
        stats[k] = band_stats(freqs, psd_all[k])
    for k, v in steps_out.items():
        psd_all[k] = periodogram_avg(v)
        stats[k] = band_stats(freqs, psd_all[k])
    # v2 终点（gc_input 全量）与 v1 终点（gc_input，K=1 seed 1235 口径）对照
    for tag, rid in (("gc_v1", "main-01"), ("gc_v2", "v2d62-main-01")):
        d = np.load(Path(cfg.outputs_root) / rid / "eval" / "gc_input.npz")
        psd_all[tag] = periodogram_avg(d["gen"])
        stats[tag] = band_stats(freqs, psd_all[tag])
        psd_all["real_full"] = periodogram_avg(d["real"])   # 同取（两次覆盖同值）
        stats["real_full"] = band_stats(freqs, psd_all["real_full"])

    # 低通 MAE 检验（全量）
    lp = {}
    for rid in ("main-01", "v2d62-main-01"):
        d = np.load(Path(cfg.outputs_root) / rid / "eval" / "gc_input.npz")
        gen, real = d["gen"], d["real"]
        m0 = float(np.abs(gen - real).mean())
        m30 = lowpass_mae(gen, real, 30.0)
        m40 = lowpass_mae(gen, real, 40.0)
        lp[rid] = {"mae_raw": m0, "mae_lp30": m30, "mae_lp40": m40,
                   "d30%": (m30 - m0) / m0 * 100, "d40%": (m40 - m0) / m0 * 100}

    # v4 滤波事实（npz filter 字段）
    filt = None
    try:
        mf = None
        import csv
        with open(os.path.join(cfg.data_root, "manifest.csv")) as f:
            mf = next(csv.DictReader(f))
        sid = mf["subject_id"]
        z = np.load(os.path.join(cfg.data_root, mf["group"], f"{sid}_rest_v4.npz"),
                    allow_pickle=True)
        filt = {"subject": sid, "filter": str(z["filter"]) if "filter" in z else None,
                "pipeline": str(z["pipeline"]) if "pipeline" in z else None}
    except Exception as e:  # noqa: BLE001
        filt = {"error": str(e)}

    res = {"elements": stats, "lowpass_mae": lp, "v4_filter_field": filt,
           "freq_res_hz": FREQ_RES, "n_windows": int(args.n_windows),
           "n_sub_steps": int(args.n_sub_steps), "diag_seed": DIAG_SEED}
    with open(out_dir / "diag_hf.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    np.savez_compressed(out_dir / "diag_hf.npz", freqs=freqs,
                        **{f"psd_{k}": v for k, v in psd_all.items()})
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
