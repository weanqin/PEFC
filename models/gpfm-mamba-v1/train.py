#!/usr/bin/env python3
"""train.py — gpfm-mamba-v1 训练循环（design §2.2/§5；R9 监控/断点续传）。

- 路径：x_t = (1−t)x0 + t·x1 + σ_min·η；t = sigmoid(z), z~N(−0.8, 0.8²) 逐窗 clamp [1e-4, 1−1e-4]（不用 U[0,1]）
- 一步生成：x̂1 = x_t + (1−t)·v_θ（训练全程不做 ODE 积分）
- 损失：1·mae + 1·cons + 0.1·tv + 0.1·corr（仅 future 128 列）
- R9：每 step 进度行（四项+总损失、it/s、ETA、lr）；每 epoch 追加 logs/metrics.jsonl；
  best = 内部验证总 loss（固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 eps 确定性复算）创新优覆写；
  last 每 epoch 无条件覆写（均含 model+EMA+optimizer+scheduler+epoch/step+RNG+best+config hash）；
  --resume 全量恢复（含 metrics.jsonl 截齐到断点 epoch）
- 早停 default-off 钩子（用户 2026-10-16 拍板，v4-B 启用 patience=10）：config 字段
  `early_stop_patience`（getattr 透传，先例 prior_bandlimit_hz/split_manifest_override）——
  best 判据连续 patience 个 epoch 无新 best 即停（break 前该 epoch best/last checkpoint 与
  metrics.jsonl 已落盘，触发行附 "early_stop": true）；v1/v2/v3 config 无此字段 ⇒ 0=禁用恒不触发。
用法：
  python3 models/gpfm-mamba-v1/train.py --smoke                 # 冒烟（4 受试者 1 epoch）
  python3 models/gpfm-mamba-v1/train.py --epochs 1 --bench      # 全量 1-epoch 基准
  python3 models/gpfm-mamba-v1/train.py --resume <last.pt>      # 断点续传（同一 Run 延续）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402
import data as D  # noqa: E402
from gpr_prior import GPRPrior  # noqa: E402
from losses import composite_loss, per_window_mae  # noqa: E402
from model import EMA, VelocityNetwork  # noqa: E402

TREND_EVERY = 5  # 每 5 step 记一次损失走势（进 metrics.jsonl，供基准四项损失走势）


class Logger:
    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="小子集冒烟（每类 2 例受试者，epochs 默认 1）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit-train-subjects", type=int, default=None, help="训练折每类受试者上限（冒烟=2）")
    p.add_argument("--resume", type=str, default=None, help="checkpoint 路径（last.pt）")
    p.add_argument("--run-id", type=str, default=None, help="指定 run 目录名（resume 默认沿用 ckpt 所属 run）")
    p.add_argument("--bench", action="store_true", help="基准模式标记（summary.json 记 100-epoch 外推）")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_t(B: int, cfg: C.Config, device) -> torch.Tensor:
    """t = sigmoid(z), z~N(t_loc, t_scale²)，逐窗 clamp（logit-normal，非 U[0,1]）。"""
    z = torch.randn(B, device=device) * cfg.t_scale + cfg.t_loc
    return torch.sigmoid(z).clamp(cfg.t_clamp_min, cfg.t_clamp_max)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/root/npi", text=True).strip()
    except Exception:
        return "unknown"


def should_early_stop(epochs_since_best: int, patience: int) -> bool:
    """早停判定（纯函数；用户 2026-10-16 拍板 v4-B 口径，default-off 钩子配套）。

    - patience <= 0：恒 False（禁用语义——v1/v2/v3 config 无 early_stop_patience 字段即禁用，
      训练行为零漂移）；
    - patience > 0：连续 epochs_since_best 个 epoch 无新 best 且达到 patience 即触发
      （恰好第 patience 轮无改善触发）。
    """
    return patience > 0 and epochs_since_best >= patience


def replay_epochs_since_best(metrics_lines) -> int:
    """从 metrics.jsonl 行序列回放早停计数（--resume 兼容，实现最省且可靠的一种）。

    每行 "best" 标志即当 epoch 的 is_best：计数 = 最近一次 best=true 之后的 epoch 数。
    checkpoint 已存 best 指标与 epoch 但不存计数——截齐后的 metrics.jsonl 与断点 epoch
    严格一致（resume 块保证），直接回放即得连续计数，无须扩展 checkpoint 结构（旧 ckpt 兼容）。
    行缺失 "best" 键（极旧行）按非 best 计。空序列 → 0。
    """
    es = 0
    for line in metrics_lines:
        if line.strip():
            es = 0 if json.loads(line).get("best") else es + 1
    return es


def evaluate_internal_val(
    model, ema, prior, val_ds, cfg, device, log
) -> Dict[str, float]:
    """best 判据：固定 t 网格 + 固定 eps 的确定性复算（跨 epoch 严格可比）。

    每个 batch 重置同一 generator（val_eps_seed）：eps/η 序列与 batch 顺序每 epoch 完全一致。
    """
    was_training = model.training
    model.eval()
    use_ema_ctx = ema.swap(model) if cfg.use_ema_for_val else None
    from contextlib import nullcontext

    with (use_ema_ctx or nullcontext()):
        loader = D.make_loader(val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed, epoch=0, num_workers=0)
        sums = {k: 0.0 for k in ("mae", "cons", "tv", "corr", "total", "xhat1_std_ema", "x1_std")}
        with torch.no_grad():
            for xp, x1 in loader:
                xp = xp.to(device, non_blocking=True)
                x1 = x1.to(device, non_blocking=True)
                B = xp.shape[0]
                g = torch.Generator(device=device)
                g.manual_seed(cfg.val_eps_seed)
                eps0 = torch.randn(B, cfg.n_channels, cfg.future_len, device=device, generator=g)
                eta = torch.randn(B, cfg.n_channels, cfg.future_len, device=device, generator=g) * cfg.sigma_min
                for tv in cfg.val_t_grid:
                    t = torch.full((B,), float(tv), device=device)
                    t3 = t[:, None, None]
                    x0 = prior.sample(xp, eps=eps0)
                    x_t = (1 - t3) * x0 + t3 * x1 + eta
                    v = model(xp, x_t, t)
                    xhat1 = x_t + (1 - t3) * v
                    parts = composite_loss(
                        xhat1, x1, cfg.w_mae, cfg.w_cons, cfg.w_tv, cfg.w_corr, cfg.corr_eps
                    )
                    # 网格均值口径（design §2.3）：对 t 网格与验证集双重平均
                    w = B / (len(val_ds) * len(cfg.val_t_grid))
                    for k in ("mae", "cons", "tv", "corr", "total"):
                        sums[k] += float(parts[k]) * w
                    sums["xhat1_std_ema"] += float(xhat1.std(dim=-1, unbiased=True).mean()) * w
                    sums["x1_std"] += float(x1.std(dim=-1, unbiased=True).mean()) * w
    if was_training:
        model.train()
    log(f"[val] 内部验证（EMA={cfg.use_ema_for_val}，t 网格 {list(cfg.val_t_grid)}（均值口径），确定性 eps seed={cfg.val_eps_seed}）"
        f" total {sums['total']:.4f} | mae {sums['mae']:.4f} cons {sums['cons']:.4f} "
        f"tv {sums['tv']:.4f} corr {sums['corr']:.4f}"
        f" | x̂1 std(EMA) {sums['xhat1_std_ema']:.4f} vs x1 std {sums['x1_std']:.4f}")
    return sums


def save_ckpt(path, model, ema, opt, sched, epoch, gs, best_val, cfg) -> None:
    torch.save(
        {
            "config_hash": cfg.config_hash(),
            "variant": cfg.variant_name,
            "epoch": epoch,
            "global_step": gs,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "best_val": best_val,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    overrides: Dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    cfg = C.build_config(**overrides)
    limit = args.limit_train_subjects
    if args.smoke:
        if limit is None:
            limit = 2  # 每类 2 例 → 训练小子集合计 4 例
        if args.epochs is None:
            cfg.epochs = 1

    device = "cuda"
    assert torch.cuda.is_available(), "训练需 CUDA（mamba_ssm）"
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    set_seed(cfg.seed)

    # ---- run 目录（resume 沿用同一 Run）----
    if args.run_id:
        run_name = args.run_id
    elif args.resume:
        run_name = Path(args.resume).parents[1].name
    else:
        run_name = time.strftime("run_%Y%m%d-%H%M%S")
    run_dir = Path(cfg.outputs_root) / run_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = Logger(str(run_dir / "logs" / "train.log"))
    tag = "smoke" if args.smoke else ("bench" if args.bench else "train")
    log(f"=== gpfm-mamba-v1 {tag} | run {run_dir} | commit {git_commit()} ===")

    # ---- resolved config / 环境快照 / z-score 统计落盘（复现三要素）----
    cfg.dump_yaml(str(run_dir / "config.yaml"))
    with open(run_dir / "env_snapshot.txt", "w") as f:
        try:
            f.write(subprocess.check_output(["pip", "freeze"], text=True))
        except Exception as e:  # noqa: BLE001
            f.write(f"pip freeze failed: {e}\n")

    # ---- 数据（全部预处理在 dataloader；z-score 仅训练折）----
    t0 = time.time()
    ds = D.build_datasets(cfg, limit_train_per_group=limit, include_test=False)
    data_load_s = time.time() - t0
    tr_ds, va_ds = ds["train"], ds["val"]
    np.save(run_dir / "channel_mean.npy", ds["channel_mean"])
    np.save(run_dir / "channel_std.npy", ds["channel_std"])
    log(f"[data] subject_filter={cfg.subject_filter} 过滤后 fold0：train {len(ds['train_entries'])} 例 {len(tr_ds)} 窗 | "
        f"val {len(ds['val_entries'])} 例 {len(va_ds)} 窗 | z-score 仅训练折 | 加载 {data_load_s:.1f}s")

    # ---- 先验 / 模型 / 优化 ----
    prior = GPRPrior(
        np.load(f"{cfg.data_root}/chan_coords_62.npy"),
        past_len=cfg.past_len, future_len=cfg.future_len,
        kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
        iso=cfg.iso, pp_jitter=cfg.pp_jitter,
        sigma_s=cfg.sigma_s, jitter_s=cfg.jitter_s,
        bandlimit_hz=getattr(cfg, "prior_bandlimit_hz", None),  # v3 变体启用；v1/v2 配置无此字段→None
    ).to(device)
    model = VelocityNetwork(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
    ).to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    log(f"[model] params {model.num_params()} | EMA {cfg.ema_decay} | AdamW lr {cfg.lr} wd {cfg.weight_decay} "
        f"cosine T_max {cfg.epochs} | clip {cfg.grad_clip} | batch {cfg.batch_size} | seed {cfg.seed}")

    # ---- resume（全量恢复：model/EMA/opt/sched/epoch/step/RNG/metrics 截齐）----
    start_epoch, gs, best_val = 0, 0, float("inf")
    epochs_since_best = 0  # 早停计数（连续无新 best 的 epoch 数；resume 时从 metrics.jsonl 回放恢复，见下）
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert ck["config_hash"] == cfg.config_hash(), (
            f"config 不一致：ckpt {ck['config_hash']} vs 当前 {cfg.config_hash()}（resume 须同 config）"
        )
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        ema.to(device)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        start_epoch = ck["epoch"] + 1
        gs = ck["global_step"]
        best_val = ck["best_val"]
        # metrics.jsonl 截齐到断点 epoch（防 kill 与写盘竞态留下重复行）
        if metrics_path.exists():
            kept = []
            for line in metrics_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec["epoch"] <= ck["epoch"]:
                        kept.append(line)
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
            # 早停计数恢复：截齐后的 metrics.jsonl 与断点 epoch 严格一致 → 回放最近一次
            # 新 best 之后的 epoch 数（replay_epochs_since_best；断点续传后早停计数连续）
            epochs_since_best = replay_epochs_since_best(metrics_path.read_text().splitlines())
        log(f"[resume] {args.resume} | epoch {ck['epoch']} -> {start_epoch} | global_step {gs} | "
            f"best_val {best_val:.4f} | RNG(python/numpy/torch/cuda)+optimizer+scheduler 已恢复 | "
            f"metrics.jsonl 截齐至 epoch {ck['epoch']}（同一 Run 延续，artifacts-management 口径）")

    # ---- 训练循环 ----
    torch.cuda.reset_peak_memory_stats()
    epoch_times = []
    total_steps_run = 0
    ck_best = run_dir / "checkpoints" / f"{cfg.variant_name}-best.pt"
    ck_last = run_dir / "checkpoints" / f"{cfg.variant_name}-last.pt"
    for epoch in range(start_epoch, cfg.epochs):
        te0 = time.time()
        loader = D.make_loader(tr_ds, cfg.batch_size, shuffle=True, seed=cfg.seed, epoch=epoch, num_workers=cfg.num_workers)
        model.train()
        acc = {k: 0.0 for k in ("mae", "cons", "tv", "corr", "total", "v_norm", "xhat1_std")}
        bucket_sum = [0.0] * 10
        bucket_cnt = [0] * 10
        trend = []
        step_times = []
        n_steps = len(loader)
        for step, (xp, x1) in enumerate(loader):
            ts0 = time.time()
            xp = xp.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            B = xp.shape[0]
            t = sample_t(B, cfg, device)
            t3 = t[:, None, None]
            x0 = prior.sample(xp)                       # μ + L_s ε L_tᵀ（GPR 条件先验）
            x_t = (1 - t3) * x0 + t3 * x1 + cfg.sigma_min * torch.randn_like(x0)
            v = model(xp, x_t, t)                       # 速度场 v_θ(x_t, t, X_past)
            xhat1 = x_t + (1 - t3) * v                  # 一步生成（训练全程不积分）
            parts = composite_loss(xhat1, x1, cfg.w_mae, cfg.w_cons, cfg.w_tv, cfg.w_corr, cfg.corr_eps)
            total = parts["total"]
            if not torch.isfinite(total):
                raise RuntimeError(f"[NaN guard] epoch {epoch} step {step}: total={total} "
                                   f"parts={ {k: float(v2) for k, v2 in parts.items()} }")
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            gs += 1
            # 监控累积
            with torch.no_grad():
                w = B / max(1, len(tr_ds))
                for k in ("mae", "cons", "tv", "corr", "total"):
                    acc[k] += float(parts[k]) * w
                acc["v_norm"] += float(v.norm(dim=(1, 2)).mean()) * w
                acc["xhat1_std"] += float(xhat1.std(dim=-1, unbiased=True).mean()) * w
                bi = (t * 10).clamp(0, 9).long()
                pm = per_window_mae(xhat1, x1)
                for i in range(10):
                    m = bi == i
                    if m.any():
                        bucket_sum[i] += float(pm[m].sum())
                        bucket_cnt[i] += int(m.sum())
            torch.cuda.synchronize()
            step_times.append(time.time() - ts0)
            total_steps_run += 1
            if step % cfg.log_every == 0:
                its = 1.0 / max(np.mean(step_times[-20:]), 1e-9)
                eta = (n_steps - step - 1) / max(its, 1e-9)
                pf = {k: float(parts[k].detach()) for k in ("mae", "cons", "tv", "corr", "total")}
                log(f"[ep {epoch + 1:03d}/{cfg.epochs} step {step + 1:04d}/{n_steps}] "
                    f"total {pf['total']:.4f} mae {pf['mae']:.4f} "
                    f"cons {pf['cons']:.4f} tv {pf['tv']:.4f} "
                    f"corr {pf['corr']:.4f} | {its:.1f} it/s ETA {eta:.0f}s "
                    f"lr {sched.get_last_lr()[0]:.2e}")
            if step % TREND_EVERY == 0:
                trend.append({"step": gs, **{k: float(parts[k].detach()) for k in ("mae", "cons", "tv", "corr", "total")}})
        sched.step()
        epoch_time = time.time() - te0
        epoch_times.append(epoch_time)

        # ---- 每 epoch：内部验证（确定性）→ best/last 覆写 → metrics.jsonl ----
        val = evaluate_internal_val(model, ema, prior, va_ds, cfg, device, log)
        is_best = val["total"] < best_val
        if is_best:
            best_val = val["total"]
        # 早停 default-off 钩子（用户 2026-10-16 拍板，v4-B 启用；先例 prior_bandlimit_hz）：
        # v1/v2/v3 config 无 early_stop_patience 字段 → getattr 默认 0 → 恒不触发（零漂移，
        # 仅多一次 getattr 与整数比较）；新 best 计数清零，否则 +1。
        _es_patience = int(getattr(cfg, "early_stop_patience", 0))
        epochs_since_best = 0 if is_best else epochs_since_best + 1
        _es_fire = should_early_stop(epochs_since_best, _es_patience)
        save_ckpt(str(ck_last), model, ema, opt, sched, epoch, gs, best_val, cfg)
        if is_best:
            save_ckpt(str(ck_best), model, ema, opt, sched, epoch, gs, best_val, cfg)
        rec = {
            "epoch": epoch,
            "global_step": gs,
            "lr": sched.get_last_lr()[0],
            "epoch_time_s": round(epoch_time, 2),
            "train": {k: round(acc[k], 5) for k in ("mae", "cons", "tv", "corr", "total")},
            "val": {k: round(v2, 5) for k, v2 in val.items()},
            "v_norm_mean": round(acc["v_norm"], 4),
            "xhat1_std_t": round(acc["xhat1_std"], 4),
            "t_bucket_mae": {f"b{i}": round(bucket_sum[i] / bucket_cnt[i], 5) if bucket_cnt[i] else None for i in range(10)},
            "t_bucket_count": bucket_cnt,
            "loss_trend": trend,
            "best": is_best,
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        }
        if _es_fire:  # 仅触发时写入（v1/v2/v3 恒不触发 → metrics 行零漂移）
            rec["early_stop"] = True
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"[epoch {epoch + 1}/{cfg.epochs} 完成] train total {acc['total']:.4f} | val total {val['total']:.4f}"
            f"{'（新 best）' if is_best else ''} | 耗时 {epoch_time:.1f}s | last{'+best' if is_best else ''} ckpt 已写")
        if _es_fire:
            # 位于每 epoch 收尾动作（best/last checkpoint + metrics.jsonl 落盘）之后——
            # 触发时该 epoch 产物完整，best/last checkpoint 语义不变（best 仍为停止前最优点）。
            log(f"[early-stop] epoch {epoch + 1}/{cfg.epochs} 触发早停：val 总 loss 连续 {epochs_since_best} 个 epoch"
                f" 无新 best（patience {_es_patience}，判据=内部验证总 loss 固定 t 网格确定性复算/EMA）| "
                f"best epoch {epoch + 1 - epochs_since_best} | best val total {best_val:.4f} | "
                f"metrics.jsonl 本 epoch 行已附 \"early_stop\": true，停止训练")
            break

    # ---- 汇总 ----
    total_train_s = float(np.sum(epoch_times)) if epoch_times else 0.0
    summary = {
        "tag": tag,
        "run_dir": str(run_dir),
        "config_hash": cfg.config_hash(),
        "code_commit": git_commit(),
        "n_train_subjects": len(ds["train_entries"]),
        "n_train_windows": len(tr_ds),
        "n_val_subjects": len(ds["val_entries"]),
        "n_val_windows": len(va_ds),
        "epochs_run": len(epoch_times),
        "batch_size": cfg.batch_size,
        "epoch_times_s": [round(x, 2) for x in epoch_times],
        "mean_epoch_time_s": round(float(np.mean(epoch_times)), 2) if epoch_times else None,
        "it_per_s": round(total_steps_run / total_train_s, 2) if total_train_s > 0 else None,
        "samples_per_s": round(len(tr_ds) * len(epoch_times) / total_train_s, 1) if total_train_s > 0 else None,
        "data_load_s": round(data_load_s, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        "best_val_total": round(best_val, 5) if best_val != float("inf") else None,
        "extrapolation_100_epochs_h": round(total_train_s / max(len(epoch_times), 1) * 100 / 3600, 3) if epoch_times else None,
        "deterministic": {"cudnn_deterministic": cfg.cudnn_deterministic, "cudnn_benchmark": cfg.cudnn_benchmark,
                          "seed": cfg.seed, "split_seed": cfg.split_seed, "val_eps_seed": cfg.val_eps_seed},
    }
    with open(run_dir / "logs" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"[summary] {json.dumps(summary, ensure_ascii=False)}")
    log("=== 训练循环结束 ===")


if __name__ == "__main__":
    main()
