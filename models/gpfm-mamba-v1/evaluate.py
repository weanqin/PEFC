#!/usr/bin/env python3
"""evaluate.py — 前置验证评估（design §6；implement.md 6.2，R9 留存）。

主指标唯一：测试折 MAE，**ODE 终点样本 x̂(1) 口径**——每窗 K=16 条 Euler 样本取均值作点预测
（勿用训练期 x̂1 = x_t+(1−t)v 公式）。
次指标：样本 CRPS（mean|x−y| − 0.5·mean|x−x′|，K² 成对含对角零的 NRG 估计）、逐通道 Pearson r 均值。
基线同口径对比：持久性（past 末值复制 128 步）、GPR 后验均值（x̂ = A·y^p）。
R1 触发监控：高 t 桶 [0.8,1] mae / 中 t 桶 [0.4,0.6] mae ≥ 1.5（训练 metrics.jsonl 末期 5 epoch
加权），或采样终点 MAE ≥ 训练末期 x̂1-MAE 的 1.5× → 告警行（口径注记：后者为 test vs train 折
不同数据，按 design §6 规定执行并如实标注）。
全部进度与结构化结果落 run 目录 logs/eval.log + logs/eval_metrics.jsonl；数值汇总 eval/summary.json；
图经 common/visualizer 落 figs/；GC/PSD/空间归因由 gc_psd.run_all 完成（--skip-gc 跳过）。

用法（真实评估，主会话在训练完成后执行）：
  python3 models/gpfm-mamba-v1/evaluate.py --run-id main-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # 仓库根 -> common/

import config as C  # noqa: E402
import gc_psd as G  # noqa: E402
from common.visualizer import Visualizer  # noqa: E402
from sample import EulerSampler, build_eval_prior, iterate_test_batches, load_eval_model  # noqa: E402

R1_RATIO = 1.5        # R1 触发阈值（design §9.1）
LATE_EPOCHS = 5       # 「训练末期」= 最后 5 epoch 均值
TRAJ_WINDOWS = 6      # 人工视觉检验素材：窗数
TRAJ_CHANNELS = 6     # 通道数
TRAJ_SAMPLES = 6      # 每窗展示样本条数（<=K）
TRAJ_CONTEXT = 96     # past 上下文点数


class Logger:
    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")


class JsonlWriter:
    def __init__(self, path: str):
        self.path = path

    def write(self, rec: Dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


# ---------------- 指标（torch，流式累积） ----------------

def channel_r(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐 (窗,通道) Pearson r（和式定义 + 近平坦置 0，与训练 corr 同守护口径），返回 (B,C)。"""
    pe = pred - pred.mean(dim=-1, keepdim=True)
    ye = y - y.mean(dim=-1, keepdim=True)
    s_xy = (pe * ye).sum(-1)
    denom = ((pe * pe).sum(-1) * (ye * ye).sum(-1)).sqrt()
    r = (s_xy / (denom + eps)).clamp(-1, 1)
    return torch.where(denom < eps, torch.zeros_like(r), r)


def crps_ensemble(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """样本 CRPS（NRG 成对近似）：mean_k|x_k−y| − 0.5·mean_{k,k'}|x_k−x_k'|。

    成对项用排序恒等式 ΣΣ|x_i−x_j| = 2·Σ_k (2k−K+1)·x_(k)（0 起始升序）向量化，
    均值分母 K²（对角为零贡献），与「numpy 公式」逐点等价。返回 (B,C,F)。
    """
    K = samples.shape[1]
    term1 = (samples - y.unsqueeze(1)).abs().mean(dim=1)
    ss, _ = torch.sort(samples, dim=1)
    w = (2.0 * torch.arange(K, device=samples.device, dtype=samples.dtype) - K + 1.0)
    pairwise = 2.0 / (K * K) * (ss * w.reshape(1, K, 1, 1)).sum(dim=1)
    return term1 - 0.5 * pairwise


class MetricAcc:
    """批加权累积（按窗数加权 → 全测试折均值口径）。"""

    def __init__(self, n_ch: int):
        self.n = 0
        self.mae = {"model": 0.0, "persistence": 0.0, "gpr_mean": 0.0}
        self.crps = 0.0
        self.r_sum = {"model": np.zeros(n_ch), "persistence": np.zeros(n_ch),
                      "gpr_mean": np.zeros(n_ch)}
        self.gen_std_t = 0.0
        self.x1_std_t = 0.0
        self.spread = 0.0  # 样本间 std（多样本展宽 vs GT 监控）

    def update(self, pred_m: torch.Tensor, pred_p: torch.Tensor, pred_g: torch.Tensor,
               samples: torch.Tensor, y: torch.Tensor) -> None:
        B = y.shape[0]
        w = B
        self.n += B
        self.mae["model"] += float((pred_m - y).abs().mean()) * B
        self.mae["persistence"] += float((pred_p - y).abs().mean()) * B
        self.mae["gpr_mean"] += float((pred_g - y).abs().mean()) * B
        self.crps += float(crps_ensemble(samples, y).mean()) * B
        for name, pred in (("model", pred_m), ("persistence", pred_p), ("gpr_mean", pred_g)):
            self.r_sum[name] += channel_r(pred, y).sum(dim=0).cpu().numpy()
        self.gen_std_t += float(samples.std(dim=-1, unbiased=True).mean()) * B
        self.x1_std_t += float(y.std(dim=-1, unbiased=True).mean()) * B
        self.spread += float(samples.std(dim=1).mean()) * B


# ---------------- R1 触发监控 ----------------

def r1_monitor(run_dir, endpoint_mae: float, log) -> Dict:
    """读训练 metrics.jsonl 末期 5 epoch：高/中 t 桶 mae 比值 + 终点 MAE 对比（design §9.1）。"""
    from pathlib import Path
    run_dir = Path(run_dir)
    mpath = run_dir / "logs" / "metrics.jsonl"
    if not mpath.exists():
        log("[R1] metrics.jsonl 缺失，跳过 R1 监控")
        return {"available": False}
    recs: List[Dict] = [json.loads(l) for l in mpath.read_text().splitlines() if l.strip()]
    late = recs[-LATE_EPOCHS:]
    hi_s = sum(r["t_bucket_mae"]["b8"] * r["t_bucket_count"][8] +
               r["t_bucket_mae"]["b9"] * r["t_bucket_count"][9] for r in late if
               r["t_bucket_mae"]["b8"] is not None and r["t_bucket_mae"]["b9"] is not None)
    hi_n = sum(r["t_bucket_count"][8] + r["t_bucket_count"][9] for r in late)
    mid_s = sum(r["t_bucket_mae"]["b4"] * r["t_bucket_count"][4] +
                r["t_bucket_mae"]["b5"] * r["t_bucket_count"][5] for r in late)
    mid_n = sum(r["t_bucket_count"][4] + r["t_bucket_count"][5] for r in late)
    train_mae_late = float(np.mean([r["train"]["mae"] for r in late]))
    out = {"available": True, "late_epochs": [r["epoch"] for r in late],
           "train_mae_late": round(train_mae_late, 5)}
    warn = []
    if hi_n > 0 and mid_n > 0:
        ratio = (hi_s / hi_n) / (mid_s / mid_n)
        out["t_bucket_ratio_high_mid"] = round(float(ratio), 4)
        if ratio >= R1_RATIO:
            warn.append(f"[R1 告警] 高 t 桶/中 t 桶 mae = {ratio:.3f} ≥ {R1_RATIO}（高 t 区欠约束风险）")
    if train_mae_late > 0:
        rel = endpoint_mae / train_mae_late
        out["endpoint_vs_train_late"] = round(float(rel), 4)
        if rel >= R1_RATIO:
            warn.append(f"[R1 告警] 采样终点 MAE {endpoint_mae:.4f} ≥ 训练末期 x̂1-MAE "
                        f"{train_mae_late:.4f} × {R1_RATIO}（口径注记：test 折终点 vs train 折一步）")
    out["triggered"] = bool(warn)
    for w in warn:
        log(w)
    if not warn:
        log(f"[R1] 未触发（比值与终点对比均 < {R1_RATIO}；详见 eval_metrics.jsonl）")
    return out


# ---------------- 主流程 ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=str, default=None,
                   help="run 目录名（默认：outputs 下最新含 best ckpt 的 run）")
    p.add_argument("--ckpt", type=str, default=None, help="显式 checkpoint 路径（默认 best）")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=None, help="每窗样本数 K（默认 cfg.n_samples=16）")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="采样 seed（默认 cfg.seed=1234；不修改 config 以免 hash 漂移）")
    p.add_argument("--skip-gc", action="store_true", help="跳过 GC/PSD/空间归因（调试用）")
    p.add_argument("--gc-bootstrap", type=int, default=1000, help="GC 相关 CI bootstrap 次数（登记）")
    return p.parse_args()


def latest_run_with_ckpt(cfg: C.Config) -> str:
    root = Path(cfg.outputs_root)
    cands = [d for d in root.iterdir()
             if d.is_dir() and (d / "checkpoints" / f"{cfg.variant_name}-best.pt").exists()]
    if not cands:
        raise FileNotFoundError(f"{root} 下无含 best checkpoint 的 run 目录")
    return max(cands, key=lambda d: d.stat().st_mtime).name  # 按修改时间取最新 run


def main() -> None:
    args = parse_args()
    cfg = C.build_config()
    device = "cuda"
    assert torch.cuda.is_available(), "评估需 CUDA（mamba_ssm）"
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    run_id = args.run_id or latest_run_with_ckpt(cfg)
    run_dir = Path(cfg.outputs_root) / run_id
    ckpt_path = args.ckpt or str(run_dir / "checkpoints" / f"{cfg.variant_name}-best.pt")
    log = Logger(str(run_dir / "logs" / "eval.log"))
    jsonl = JsonlWriter(str(run_dir / "logs" / "eval_metrics.jsonl"))
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    K = args.n_samples or cfg.n_samples
    sample_seed = args.sample_seed if args.sample_seed is not None else cfg.seed

    log(f"=== evaluate | run {run_dir} | ckpt {ckpt_path} | K={K} euler={cfg.euler_steps} "
        f"seed={sample_seed}（GC 单样本 seed={sample_seed + 1}）===")
    model, ema, ck = load_eval_model(ckpt_path, cfg, device)
    prior = build_eval_prior(cfg, device)
    log(f"[ckpt] epoch {ck['epoch']} | best_val {ck['best_val']:.4f} | config_hash "
        f"{ck['config_hash']}（已校验一致）| 评估用 EMA 权重（decay {ck['ema']['decay']}）")
    loader, test_ds = iterate_test_batches(cfg, run_dir, args.batch_size)
    n_win = len(test_ds)
    log(f"[data] 测试折 {len(test_ds.subjects)} 例 {n_win} 窗 | z-score = run 目录训练折统计"
        f"（与重算 allclose 校验通过）")

    sampler = EulerSampler(model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min)
    acc = MetricAcc(cfg.n_channels)

    # 选定人工视觉检验窗/通道（确定性等距）
    traj_w = np.linspace(0, n_win - 1, TRAJ_WINDOWS).astype(int)
    traj_c = np.linspace(0, cfg.n_channels - 1, TRAJ_CHANNELS).astype(int)
    traj_store = {"past": [], "real": [], "gen": []}
    traj_set = set(int(i) for i in traj_w)

    # ---- Pass A：K 样本 → 主/次指标（流式，不落全量样本）----
    t0 = time.time()
    g = torch.Generator(device=device)
    g.manual_seed(sample_seed)
    n_done = 0
    with ema.swap(model):
        for bi, (xp, x1) in enumerate(loader):
            xp = xp.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            samples = sampler.sample(xp, K, generator=g)              # (B,K,C,F) 终点样本
            pred_m = samples.mean(dim=1)                              # 点预测 = K 样本均值（主指标口径）
            pred_p = xp[:, :, -1:].expand(-1, -1, cfg.future_len).contiguous()  # 持久性
            pred_g = prior.posterior_mean(xp)                         # GPR 后验均值
            acc.update(pred_m, pred_p, pred_g, samples, x1)
            gidx0 = n_done
            n_done += xp.shape[0]
            for wi, gidx in enumerate(range(gidx0, n_done)):
                if gidx in traj_set:
                    traj_store["past"].append(xp[wi, traj_c, -TRAJ_CONTEXT:].cpu().numpy())
                    traj_store["real"].append(x1[wi, traj_c].cpu().numpy())
                    traj_store["gen"].append(samples[wi, traj_c, :, :TRAJ_SAMPLES].cpu().numpy())
            el = time.time() - t0
            eta = el / n_done * (n_win - n_done)
            log(f"[pass-A] batch {bi + 1}/{len(loader)} | 窗 {n_done}/{n_win} | {el:.0f}s | "
                f"ETA {eta:.0f}s | model-MAE(累计) {acc.mae['model'] / acc.n:.4f}")

    res = {
        "n_windows": acc.n, "K": K, "euler_steps": cfg.euler_steps, "sample_seed": sample_seed,
        "mae": {k: round(v / acc.n, 5) for k, v in acc.mae.items()},
        "crps_model": round(acc.crps / acc.n, 5),
        "pearson_r_mean": {k: round(float(np.mean(v / acc.n)), 5) for k, v in acc.r_sum.items()},
        "pearson_r_per_channel": {k: (v / acc.n).round(4).tolist() for k, v in acc.r_sum.items()},
        "gen_std_t": round(acc.gen_std_t / acc.n, 5),
        "x1_std_t": round(acc.x1_std_t / acc.n, 5),
        "sample_spread_mean": round(acc.spread / acc.n, 5),
    }
    log(f"[结果] 主指标 test-MAE（ODE 终点样本均值口径）= {res['mae']['model']:.5f} | "
        f"持久性 {res['mae']['persistence']:.5f} | GPR 均值 {res['mae']['gpr_mean']:.5f}")
    log(f"[结果] CRPS {res['crps_model']:.5f} | r(model) {res['pearson_r_mean']['model']:.4f} | "
        f"r(persistence) {res['pearson_r_mean']['persistence']:.4f} | "
        f"r(gpr) {res['pearson_r_mean']['gpr_mean']:.4f}")
    log(f"[结果] 幅度：生成 std_t {res['gen_std_t']:.4f} vs 真实 {res['x1_std_t']:.4f} | "
        f"样本间展宽 {res['sample_spread_mean']:.4f}")
    jsonl.write({"stage": "pass_a", **res})

    # ---- R1 触发监控 ----
    r1 = r1_monitor(run_dir, res["mae"]["model"], log)
    jsonl.write({"stage": "r1", **r1})

    # ---- Pass B：GC/PSD 输入（每窗固定 seed 单样本 + 真实 future，两组同分段同拼接）----
    log("[pass-B] 生成 GC/PSD 输入（K=1，seed=%d，确定性顺序 loader）" % (sample_seed + 1))
    t0 = time.time()
    g2 = torch.Generator(device=device)
    g2.manual_seed(sample_seed + 1)
    gen_all = np.empty((n_win, cfg.n_channels, cfg.future_len), dtype=np.float32)
    real_all = np.empty_like(gen_all)
    subj_ids = np.array([e["subject_id"] for e in test_ds.subjects])
    win_subj = np.array([subj_ids[si] for si, _ in test_ds.windows], dtype=np.int64)
    win_start = np.array([s for _, s in test_ds.windows], dtype=np.int64)
    n_done = 0
    with ema.swap(model):
        for bi, (xp, x1) in enumerate(loader):
            xp = xp.to(device, non_blocking=True)
            s1 = sampler.sample(xp, 1, generator=g2)                  # (B,1,C,F)
            gen_all[n_done:n_done + xp.shape[0]] = s1[:, 0].cpu().numpy()
            real_all[n_done:n_done + xp.shape[0]] = x1.numpy()
            n_done += xp.shape[0]
            if bi % 20 == 0 or bi == len(loader) - 1:
                el = time.time() - t0
                log(f"[pass-B] batch {bi + 1}/{len(loader)} | 窗 {n_done}/{n_win} | {el:.0f}s | "
                    f"ETA {el / n_done * (n_win - n_done):.0f}s")
    gc_in = run_dir / "eval" / "gc_input.npz"
    np.savez_compressed(gc_in, gen=gen_all, real=real_all, win_subj=win_subj,
                        win_start=win_start, gc_seed=sample_seed + 1)
    log(f"[pass-B] 落盘 {gc_in}（gen/real (N,62,128) f32 + 窗元数据）")
    jsonl.write({"stage": "pass_b", "gc_input": str(gc_in), "n_windows": n_win,
                 "gc_seed": sample_seed + 1})

    # ---- 图（统一 visualizer）----
    viz = Visualizer(str(run_dir / "figs"))
    traj_path = viz.render("trajectories", {
        "past": np.stack(traj_store["past"]), "real": np.stack(traj_store["real"]),
        "gen": np.stack(traj_store["gen"]),
        "window_ids": [int(i) for i in traj_w], "channel_ids": [int(c) for c in traj_c],
        "fs": 500,
    }, f"eval_trajectories_{run_id}.png")
    metrics_recs = [json.loads(l) for l in
                    (run_dir / "logs" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    curve_path = viz.render("training_curves", {"records": metrics_recs},
                            f"eval_training_curves_{run_id}.png")
    log(f"[figs] {traj_path} | {curve_path}")
    jsonl.write({"stage": "figures", "files": [traj_path, curve_path]})

    # ---- 汇总（含 GC/PSD 若未跳过）----
    summary = {"run_id": run_id, "ckpt": str(ckpt_path), "epoch": ck["epoch"],
               "config_hash": ck["config_hash"], "pass_a": res, "r1": r1,
               "figures": [traj_path, curve_path]}
    if not args.skip_gc:
        gc_res = G.run_all(run_dir=str(run_dir), data_root=cfg.data_root,
                           sigma_s=cfg.sigma_s, jitter_s=cfg.jitter_s,
                           p_max=10, bootstrap_n=args.gc_bootstrap, fs=500.0,
                           nperseg=512, stride=cfg.stride, past_len=cfg.past_len,
                           log=log, jsonl=jsonl)
        summary["gc_psd"] = gc_res
    with open(run_dir / "eval" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    jsonl.write({"stage": "done", "summary": str(run_dir / "eval" / "summary.json")})
    log("=== evaluate 完成 ===")


if __name__ == "__main__":
    main()
