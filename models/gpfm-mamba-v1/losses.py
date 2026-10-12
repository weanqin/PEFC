"""复合损失 mae + cons + tv + corr（design §2.3 逐字；定义域 = future 128 列）。

- mae  = mean_{b,c,τ} |x̂1 − x1|（分母 B·C·T）
- cons = mean_{b,c} [ |std_τ(x̂1) − std_τ(x1)| + |mean_τ(x̂1) − mean_τ(x1)| ]（std ddof=1，两项相加，分母 B·C）
- tv   = mean_{b,c,τ} |x̂1[τ+1] − x̂1[τ]|（分母 B·C·(T−1)，纯平滑惩罚不对照 GT）
- corr = mean_{b,c} [1 − r]，r = S_xy/(√(S_xx·S_yy)+eps)（和式、eps=1e-6、截断 [−1,1]、近平坦置 r=0）
- 总 = w·各项；全部 NaN guard（nan_to_num）
"""

from __future__ import annotations

from typing import Dict

import torch


def _guard(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def mae_loss(xhat1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return _guard((xhat1 - x1).abs().mean())


def cons_loss(xhat1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    std_p = xhat1.std(dim=-1, unbiased=True)  # (B,C) ddof=1
    std_g = x1.std(dim=-1, unbiased=True)
    mean_p = xhat1.mean(dim=-1)
    mean_g = x1.mean(dim=-1)
    per_bc = (std_p - std_g).abs() + (mean_p - mean_g).abs()  # 两项相加
    return _guard(per_bc.mean())


def tv_loss(xhat1: torch.Tensor) -> torch.Tensor:
    return _guard((xhat1[..., 1:] - xhat1[..., :-1]).abs().mean())


def corr_loss(xhat1: torch.Tensor, x1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xe = xhat1 - xhat1.mean(dim=-1, keepdim=True)
    xg = x1 - x1.mean(dim=-1, keepdim=True)
    s_xy = (xe * xg).sum(dim=-1)
    s_xx = (xe * xe).sum(dim=-1)
    s_yy = (xg * xg).sum(dim=-1)
    denom = (s_xx * s_yy).sqrt() + eps
    r = (s_xy / denom).clamp(-1.0, 1.0)
    r = torch.where((s_xx < eps) | (s_yy < eps), torch.zeros_like(r), r)  # 近平坦通道置 r=0
    return _guard((1.0 - r).mean())


def composite_loss(
    xhat1: torch.Tensor,
    x1: torch.Tensor,
    w_mae: float = 1.0,
    w_cons: float = 1.0,
    w_tv: float = 0.1,
    w_corr: float = 0.1,
    corr_eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    mae = mae_loss(xhat1, x1)
    cons = cons_loss(xhat1, x1)
    tv = tv_loss(xhat1)
    corr = corr_loss(xhat1, x1, eps=corr_eps)
    total = w_mae * mae + w_cons * cons + w_tv * tv + w_corr * corr
    return {"mae": mae, "cons": cons, "tv": tv, "corr": corr, "total": _guard(total)}


def per_window_mae(xhat1: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """按 t 分桶监控用：每窗 mae (B,)。"""
    return (xhat1 - x1).abs().mean(dim=(1, 2))
