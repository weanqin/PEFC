"""response.py — CRN 批采样 + 响应度量 D + null 距离（design §2.2/§2.3；B0 骨架，函数级单测）。

- **CRN 契约**：逐窗**一组** eps/eta，跨 baseline 与全部 (A, r) 变体共享（预注册钉死）；
  显式 eps/eta 下 `EulerSampler.sample` 完全确定（无内部 RNG 消耗）。
- **响应度量** D_{A→B}(r;w) = RMS_{k,t}(δ) / s_{w,B}：δ = 同 k 样本 perturbed − baseline
  配对差；s = baseline 生成分布在靶通道的自身波动尺度（纯模型侧，不依赖真实未来段）。
- **null**（§2.3）：同窗 baseline K 条样本两两配对同构距离（与 D 同构、扰动为零），
  τ 逐被试分池分位（B1 主 Run 落盘聚合，本模块只供距离函数）。
"""

from __future__ import annotations

import torch

import perturb as PB


def crn_noise(win_idx: int, n_samples: int, n_channels: int, future_len: int,
              device, seed: int = None):
    """逐窗一组 eps/eta（(1, K, C, F)，跨全变体与 baseline 共享——design §2.2 预注册钉死）。

    确定性方案：`torch.Generator(device)` manual_seed(SEED + win_idx)，先 eps 后 eta；
    win_idx = 主 Run 内全局窗序号（跨例连续，迭代顺序 B1 登记 experiment.md）。
    """
    PC = PB.load_config()
    g = torch.Generator(device=device)
    g.manual_seed((PC.SEED if seed is None else seed) + win_idx)
    eps = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    eta = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    return eps, eta


@torch.no_grad()
def sample_window_variants(sampler, x_past: torch.Tensor, win_idx: int,
                           seed: int = None) -> torch.Tensor:
    """单窗 CRN 批采样（design §2.2 批采样纪律）：**一次** `sample()` 调用，baseline 与
    全部 (A, r) 变体并入同一 batch (1+62×8, C, P)，K 条样本共享同组 eps/eta。

    x_past (C, P) 或 (1, C, P)；返回 samples (1+62×8, K, C, F)（行 0 = baseline）。
    """
    PC = PB.load_config()
    x = x_past.unsqueeze(0) if x_past.dim() == 2 else x_past
    assert x.dim() == 3 and x.shape[0] == 1, f"单窗接口，输入形状 {tuple(x.shape)}"
    eps, eta = crn_noise(win_idx, PC.K, x.shape[1], sampler.prior.future_len,
                         x.device, seed)
    batch = PB.build_variant_batch(x)
    return sampler.sample(batch, PC.K, eps=eps, eta=eta)


def baseline_scale(base: torch.Tensor) -> torch.Tensor:
    """s_{w,B} = RMS_{k,t}(x0 − \bar{k})（design §2.2）：base (K, C, F) → (C,)。

    \bar{k} = 逐 (B, t) 对 k 求均值（baseline 生成分布靶通道自身波动尺度）。
    """
    dev = base - base.mean(dim=0, keepdim=True)
    return dev.pow(2).mean(dim=(0, 2)).sqrt()


def d_curves(samples: torch.Tensor) -> torch.Tensor:
    """窗级 D 曲线（design §2.2）：samples (1+62×8, K, C, F) → D (62, 8, 62)。

    输出索引 [源 A, r 档 j, 靶 B]（行=源、列=靶方向约定）；变体序 = perturb.variant_index
    （a 主序 / j 次序）。r=0 点不在采样批内（CRN 恒等按构造取 D(0)=0，主曲线置 0）。
    """
    PC = PB.load_config()
    base = samples[0]
    s = baseline_scale(base)
    delta = samples[1:] - base                                  # (62×8, K, C, F)
    d = delta.pow(2).mean(dim=(1, 3)).sqrt() / s                # (62×8, C)
    return d.view(PC.N_SRC, PC.N_R_PERT, -1)


def null_distances(base: torch.Tensor) -> torch.Tensor:
    """§2.3 null：同窗 baseline K 条两两配对 d(i,j) = RMS_t(x_i − x_j)/s_B。

    base (K, C, F) → (K(K−1)/2, C)；K=8 → 28 对/窗/通道（τ 分池样本源，B1 主 Run 复用）。
    """
    n = base.shape[0]
    s = baseline_scale(base)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((base[i] - base[j]).pow(2).mean(dim=-1).sqrt() / s)
    return torch.stack(rows)
