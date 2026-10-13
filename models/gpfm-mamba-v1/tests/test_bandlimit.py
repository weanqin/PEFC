#!/usr/bin/env python3
"""test_bandlimit.py — GPRPrior 采样后带限（Run 015/016，v3 变体）单测。

1) bandlimit_hz=None：sample 与手工装配逐位一致、bandlimit 恒等（v1/v2 行为回归守护）
2) bandlimit_hz=40：样本谱 >40 Hz 功率占比 ≈0（对照未滤波显著非零）
3) 带限后同窗先验散布保留 ≥80%（Run 015 探针 ~95%）
4) posterior_mean 不受 bandlimit 开关影响（GPR 基线口径可比）
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_prior import GPRPrior  # noqa: E402

DATA = "/root/autodl-tmp/edra_process_v4"


def psd_hi_frac(x: torch.Tensor) -> float:
    w = np.hanning(x.shape[-1])
    sp = (np.abs(np.fft.rfft(x.numpy() * w, axis=-1)) ** 2).mean(axis=(0, 1))
    fr = np.fft.rfftfreq(x.shape[-1], d=1.0 / 500.0)
    return float(sp[fr > 40.9].sum() / sp.sum())  # 128 点分辨率 3.906 Hz：>40.9 取 42.97+ bins


def main() -> None:
    coords = np.load(f"{DATA}/chan_coords_62.npy")
    xp = torch.randn(4, 62, 384, generator=torch.Generator().manual_seed(1))
    eps = torch.randn(4, 62, 128, generator=torch.Generator().manual_seed(7))

    p0 = GPRPrior(coords, spatial="identity")                        # v1/v2 口径（带限关）
    p1 = GPRPrior(coords, spatial="identity", bandlimit_hz=40.0)     # v3 口径

    # 1) None 回归：逐位一致 + 恒等
    x0 = p0.sample(xp, eps=eps)
    manual = p0.posterior_mean(xp) + torch.einsum("ij,bjk,lk->bil", p0.L_s, eps, p0.L_t)
    assert torch.equal(x0, manual), "bandlimit_hz=None 时 sample 必须与手工装配逐位一致"
    assert torch.equal(p0.bandlimit(x0), x0), "bandlimit(None) 必须恒等返回"

    # 2) 带限谱：>40 Hz 功率塌缩（对照未滤波显著非零）
    x0f = p1.sample(xp, eps=eps)
    f_raw, f_bl = psd_hi_frac(x0), psd_hi_frac(x0f)
    print(f"[2] >40Hz 功率占比: 未滤波 {f_raw:.4f} → 带限 {f_bl:.2e}")
    assert f_bl < 1e-3, f"带限后 >40Hz 占比应 <1e-3，got {f_bl:.2e}"
    assert f_raw > 0.02, f"未滤波对照应 >2%（敏感性守护），got {f_raw:.4f}"

    # 3) 同窗散布保留（K=16 同 μ 不同噪声）
    eps16 = torch.randn(16, 62, 128, generator=torch.Generator().manual_seed(9))
    noise = torch.einsum("ij,kjc,lc->kil", p0.L_s, eps16, p0.L_t)    # (16,62,128)
    sp_raw = noise.std(dim=0).mean().item()
    sp_bl = p1.bandlimit(noise).std(dim=0).mean().item()
    print(f"[3] 同窗先验散布: {sp_raw:.3f} → {sp_bl:.3f}（保留 {sp_bl / sp_raw * 100:.0f}%）")
    assert sp_bl / sp_raw > 0.8, f"散布保留应 ≥80%，got {sp_bl / sp_raw * 100:.0f}%"

    # 4) posterior_mean 不受开关影响（GPR 基线口径）
    assert torch.equal(p0.posterior_mean(xp), p1.posterior_mean(xp)), \
        "posterior_mean 不应随 bandlimit 开关变化"

    print("test_bandlimit: 4/4 PASS")


if __name__ == "__main__":
    main()
