#!/usr/bin/env python3
"""sample.py — Euler ODE 采样（design §2.2/§5/§6.1；implement.md 6.1）。

- 采样口径（评估唯一口径，勿与训练期一步生成 x̂1 混淆）：
  从 GPR 条件先验 x0 出发（+ σ_min·ε），t: 0→1 均分 euler_steps=16 步，
  逐步 x ← x + Δt·v_θ(x, t, X_past)；终点 x̂(1) 即评估用生成样本。
- 权重：checkpoint 的 `ema` state（评估只读 ema + config_hash 校验）。
- 每窗 K 条独立样本（evaluate 调用；K=cfg.n_samples=16）；固定采样 seed（=cfg.seed，
  默认 1234；不修改 config.py 以免 config_hash 漂移——main-01 训练中以 05f7a30c448d 校验）。
- R9：逐批打印进度与 ETA。
- 依赖 config/model/data/gpr_prior，不复制逻辑。

用法（冒烟：小窗数验证机械正确性；真实评估由主会话经 evaluate.py 执行）：
  python3 models/gpfm-mamba-v1/sample.py --ckpt <best.pt> --limit-windows 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402
import data as D  # noqa: E402
from gpr_prior import GPRPrior  # noqa: E402
from model import EMA, VelocityNetwork  # noqa: E402


def load_eval_model(
    ckpt_path: str, cfg: C.Config, device: str
) -> Tuple[VelocityNetwork, EMA, dict]:
    """评估装载：只读 ema 权重 + config_hash 校验（checkpoint 契约见 train.save_ckpt）。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("config_hash") != cfg.config_hash():
        raise RuntimeError(
            f"config_hash 不一致：ckpt {ck.get('config_hash')} vs 当前 {cfg.config_hash()}——"
            "评估须与训练同 resolved config"
        )
    model = VelocityNetwork(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
    ).to(device)
    model.load_state_dict(ck["model"])  # 结构/键校验
    model.eval()
    ema = EMA(model, decay=ck["ema"]["decay"])
    ema.load_state_dict(ck["ema"])
    ema.to(device)
    return model, ema, ck


def build_eval_prior(cfg: C.Config, device: str) -> GPRPrior:
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    return GPRPrior(
        coords, past_len=cfg.past_len, future_len=cfg.future_len,
        kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
        iso=cfg.iso, pp_jitter=cfg.pp_jitter,
        sigma_s=cfg.sigma_s, jitter_s=cfg.jitter_s,
        bandlimit_hz=getattr(cfg, "prior_bandlimit_hz", None),  # v3 变体；v1/v2 无此字段→None
    ).to(device)


class EulerSampler:
    """Euler ODE 采样器（EMA 权重经 swap 上下文由调用方包住，或直接传入已 swap 模型）。

    sample(x_past, n_samples, generator, eps) -> (B, K, C, F)：
    - eps 显式传入则完全确定（单测：batch 维独立性 / 确定性）；
    - K 条样本按 (B*K) 展平批积分，chunk_bk 控制单次前向的展平批量上限（显存）。
    - 逐窗 t 同步：同一窗的 K 条样本共享同一 t 网格（每步一个 t 标量广播）。
    """

    def __init__(self, model: VelocityNetwork, prior: GPRPrior,
                 euler_steps: int = 16, sigma_min: float = 1e-4,
                 chunk_bk: int = 1024):
        self.model = model
        self.prior = prior
        self.steps = euler_steps
        self.sigma_min = sigma_min
        self.chunk_bk = chunk_bk

    @torch.no_grad()
    def sample(
        self,
        x_past: torch.Tensor,                       # (B, C, P)
        n_samples: int,
        generator: Optional[torch.Generator] = None,
        eps: Optional[torch.Tensor] = None,          # (B, K, C, F) 显式先验噪声
        eta: Optional[torch.Tensor] = None,          # (B, K, C, F) 显式 σ_min 噪声
    ) -> torch.Tensor:                              # (B, K, C, F)
        B, Cn, P = x_past.shape
        F_ = self.prior.future_len
        K = n_samples
        dev = x_past.device
        if eps is None:
            eps = torch.randn(B, K, Cn, F_, device=dev, generator=generator)
        # GPR 条件先验出发点 x0 = LP(μ + L_s ε L_tᵀ) + σ_min·η（LP=bandlimit_hz 时启用，与训练
        # prior.sample 末端同滤一致；σ_min·η 不滤——训练路径 x_t=(1−t)·LP(x0)+t·x1+σ_min·η 同口径）
        mu = self.prior.posterior_mean(x_past)                       # (B,C,F)
        x0 = self.prior.bandlimit(mu.unsqueeze(1) + torch.einsum(
            "ij,bqjk,lk->bqil", self.prior.L_s, eps, self.prior.L_t
        ))                                                            # (B,K,C,F)
        if self.sigma_min > 0:
            if eta is None:
                eta = torch.randn(B, K, Cn, F_, device=dev, generator=generator)
            x0 = x0 + self.sigma_min * eta
        dt = 1.0 / self.steps
        outs = []
        for c0 in range(0, B * K, self.chunk_bk):
            c1 = min(c0 + self.chunk_bk, B * K)
            x = x0.reshape(B * K, Cn, F_)[c0:c1].contiguous()        # (bk,C,F)
            xp = x_past.unsqueeze(1).expand(B, K, Cn, P).reshape(B * K, Cn, P)[c0:c1] \
                .contiguous()
            for i in range(self.steps):
                t_val = i * dt
                t = torch.full((x.shape[0],), t_val, device=dev, dtype=x.dtype)
                v = self.model(xp, x, t)                             # v_θ(x, t, X_past)
                x = x + dt * v
            outs.append(x)
        return torch.cat(outs, dim=0).reshape(B, K, Cn, F_)


def iterate_test_batches(cfg: C.Config, run_dir: Path, batch_size: int,
                         num_workers: int = 2):
    """测试折确定性 loader + 评估侧 z-score 口径（run 目录落盘的训练折统计，train.py 已存）。

    返回 (loader, test_ds)。统计一致性用重算 allclose 断言（防错拿 run 目录）。
    """
    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    D.fixate_split(cfg, train, val, test)  # 划分固化校验，漂移即报错
    mean = np.load(Path(run_dir) / "channel_mean.npy")
    std = np.load(Path(run_dir) / "channel_std.npy")
    mean_r, std_r = D.compute_channel_stats(train)  # 仅训练折（与训练同口径）
    assert np.allclose(mean, mean_r, rtol=1e-6, atol=1e-8), "run 目录 z-score 统计与训练折不符"
    assert np.allclose(std, std_r, rtol=1e-6, atol=1e-8), "run 目录 z-score 统计与训练折不符"
    test_ds = D.EdraWindowsDataset(
        test, mean, std, past_len=cfg.past_len, future_len=cfg.future_len, stride=cfg.stride
    )
    loader = D.make_loader(test_ds, batch_size, shuffle=False, seed=cfg.seed, epoch=0,
                           num_workers=num_workers)
    return loader, test_ds


def find_best_ckpt(cfg: C.Config, run_id: str) -> Path:
    run_dir = Path(cfg.outputs_root) / run_id
    ck = run_dir / "checkpoints" / f"{cfg.variant_name}-best.pt"
    if not ck.exists():
        raise FileNotFoundError(f"best checkpoint 不存在: {ck}")
    return ck


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run-id", type=str, default="bench-1ep-46subj")
    p.add_argument("--limit-windows", type=int, default=8, help="冒烟窗数上限（真实评估走 evaluate.py）")
    p.add_argument("--batch-size", type=int, default=4, help="冒烟小批（GPU 与训练共存约束）")
    p.add_argument("--n-samples", type=int, default=None, help="默认 cfg.n_samples")
    args = p.parse_args()

    cfg = C.build_config()
    device = "cuda"
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    ckpt_path = args.ckpt or str(find_best_ckpt(cfg, args.run_id))
    # 冒烟兼容 bench checkpoint（--epochs 1 覆写 → hash 含 epochs）：失配则按 ckpt epoch 复原该覆写
    probe = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if probe["config_hash"] != cfg.config_hash():
        cfg = C.build_config(epochs=probe["epoch"] + 1)
        print(f"[sample] hash 失配，按 ckpt epoch 复原 epochs={probe['epoch'] + 1} 覆写"
              f"（bench 口径；evaluate.py 正式评估仍走严格校验）", flush=True)
    model, ema, ck = load_eval_model(ckpt_path, cfg, device)
    prior = build_eval_prior(cfg, device)
    run_dir = Path(cfg.outputs_root) / args.run_id
    loader, test_ds = iterate_test_batches(cfg, run_dir, args.batch_size)
    K = args.n_samples or cfg.n_samples
    sampler = EulerSampler(model, prior, euler_steps=cfg.euler_steps,
                           sigma_min=cfg.sigma_min)
    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed)
    n_total = min(len(test_ds), args.limit_windows) if args.limit_windows else len(test_ds)
    t0 = time.time()
    n_done = 0
    with ema.swap(model):
        for bi, (xp, x1) in enumerate(loader):
            if n_done >= n_total:
                break
            xp = xp.to(device, non_blocking=True)
            s = sampler.sample(xp, K, generator=g)
            assert s.shape == (xp.shape[0], K, cfg.n_channels, cfg.future_len)
            assert torch.isfinite(s).all()
            n_done += xp.shape[0]
            el = time.time() - t0
            eta = el / n_done * (n_total - n_done)
            print(f"[sample] batch {bi + 1} | 窗 {n_done}/{n_total} | {el:.1f}s | "
                  f"ETA {eta:.0f}s | 输出 {tuple(s.shape)} 有限", flush=True)
    print(f"[sample] 冒烟完成（EMA 权重，euler_steps={cfg.euler_steps}，K={K}，"
          f"epoch={ck['epoch']}）——真实评估由主会话经 evaluate.py 执行")


if __name__ == "__main__":
    main()
