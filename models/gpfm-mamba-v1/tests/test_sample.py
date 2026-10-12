"""test_sample.py — Euler ODE 采样单测（implement.md 6.1）。

① 形状 (B,K,62,128) 与有限性（随机权重小模型 + 生产尺寸 bench checkpoint）
② 确定性：同 seed 同输出
③ batch 维度独立性：显式 eps/eta 下整批 vs 逐窗一致
④ checkpoint 装载路径（bench-1ep-46subj best.pt 只读）：config_hash 校验 + EMA shadow + 生产尺寸采样
运行：python3 models/gpfm-mamba-v1/tests/test_sample.py（需 CUDA；batch 保持小——与训练共存）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from gpr_prior import GPRPrior  # noqa: E402
from model import VelocityNetwork  # noqa: E402
from sample import EulerSampler, load_eval_model  # noqa: E402


def small_prior(past=64, fut=32, n_ch=8, device="cuda"):
    rng = np.random.default_rng(7)
    coords = rng.uniform(-0.1, 0.1, size=(n_ch, 3))
    return GPRPrior(coords, past_len=past, future_len=fut, kernel_t="ou", gamma=1.0,
                    period_p=50, iso=1e-2, pp_jitter=1e-4, sigma_s=0.080,
                    jitter_s=1e-3).to(device)


def main() -> None:
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    cfg = C.Config()
    dev = "cuda"
    torch.manual_seed(cfg.seed)

    B, K, n_ch, P, F = 2, 3, 8, 64, 32
    model = VelocityNetwork(n_channels=n_ch, past_len=P, future_len=F,
                            d_model=16, n_layers=2, d_state=8, d_conv=4, expand=2,
                            time_emb_dim=16).to(dev)
    model.eval()
    prior = small_prior(P, F, n_ch, dev)
    sampler = EulerSampler(model, prior, euler_steps=8, sigma_min=1e-4, chunk_bk=64)
    xp = torch.randn(B, n_ch, P, device=dev)

    # ① 形状 + 有限
    g = torch.Generator(device=dev)
    g.manual_seed(123)
    s1 = sampler.sample(xp, K, generator=g)
    assert s1.shape == (B, K, n_ch, F), f"形状错误 {s1.shape}"
    assert torch.isfinite(s1).all()
    print(f"[形状] {tuple(s1.shape)}（B={B}, K={K}, C={n_ch}, F={F}）| euler 8 步 | 全部有限")

    # ② 确定性：同 seed 同输出
    g2 = torch.Generator(device=dev)
    g2.manual_seed(123)
    s2 = sampler.sample(xp, K, generator=g2)
    maxdiff = float((s1 - s2).abs().max())
    assert torch.allclose(s1, s2, atol=1e-6), f"同 seed 输出不一致 max|Δ|={maxdiff}"
    print(f"[确定性] 同 seed max|Δ| = {maxdiff:.2e}（atol 1e-6 内）")
    g3 = torch.Generator(device=dev)
    g3.manual_seed(456)
    s3 = sampler.sample(xp, K, generator=g3)
    assert not torch.allclose(s1, s3, atol=1e-4), "不同 seed 应产生不同样本"

    # ③ batch 维度独立性：显式 eps/eta，整批 vs 逐窗切片一致
    eps = torch.randn(B, K, n_ch, F, device=dev)
    eta = torch.randn(B, K, n_ch, F, device=dev)
    s_full = sampler.sample(xp, K, eps=eps, eta=eta)
    s_w0 = sampler.sample(xp[:1], K, eps=eps[:1], eta=eta[:1])
    s_w1 = sampler.sample(xp[1:], K, eps=eps[1:], eta=eta[1:])
    d0 = float((s_full[0] - s_w0[0]).abs().max())
    d1 = float((s_full[1] - s_w1[0]).abs().max())
    assert torch.allclose(s_full[0], s_w0[0], atol=1e-4), f"窗 0 batch 依赖 max|Δ|={d0}"
    assert torch.allclose(s_full[1], s_w1[0], atol=1e-4), f"窗 1 batch 依赖 max|Δ|={d1}"
    print(f"[batch 独立] 整批 vs 逐窗 max|Δ| = {d0:.2e} / {d1:.2e}（atol 1e-4 内）")

    # ④ 生产尺寸：bench-1ep-46subj best checkpoint（已完成、只读）装载 + 单窗采样
    #    bench 用 --epochs 1 跑成 → resolved config hash 含 epochs=1，构造时须同参
    ck_path = "/root/npi/outputs/gpfm-mamba-f0/bench-1ep-46subj/checkpoints/gpfm-mamba-f0-best.pt"
    if os.path.exists(ck_path):
        cfg_bench = C.build_config(epochs=1)
        m2, ema2, ck = load_eval_model(ck_path, cfg_bench, dev)
        prior2 = GPRPrior(
            np.load(f"{cfg.data_root}/chan_coords_62.npy"),
            past_len=cfg.past_len, future_len=cfg.future_len,
            kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
            iso=cfg.iso, pp_jitter=cfg.pp_jitter,
            sigma_s=cfg.sigma_s, jitter_s=cfg.jitter_s,
        ).to(dev)
        sampler2 = EulerSampler(m2, prior2, euler_steps=cfg.euler_steps,
                                sigma_min=cfg.sigma_min, chunk_bk=256)
        xp2 = torch.randn(1, cfg.n_channels, cfg.past_len, device=dev)
        gb = torch.Generator(device=dev)
        gb.manual_seed(cfg.seed)
        with ema2.swap(m2):
            out = sampler2.sample(xp2, 2, generator=gb)
        assert out.shape == (1, 2, cfg.n_channels, cfg.future_len), f"生产尺寸形状错误 {out.shape}"
        assert torch.isfinite(out).all()
        assert set(ck["ema"]["shadow"].keys()) == set(m2.state_dict().keys()), "EMA shadow 键不齐"
        print(f"[ckpt 装载] bench best（epoch {ck['epoch']}，hash {ck['config_hash']} 校验通过）| "
              f"EMA swap 采样 {tuple(out.shape)}（62ch/384+128/euler {cfg.euler_steps}）")
    else:
        print(f"[ckpt 装载] 跳过（{ck_path} 不存在）")

    print("[PASS] test_sample 全部断言通过")


if __name__ == "__main__":
    main()
