"""官方 mamba-ssm 适配层（mamba_ssm 2.3.1 + causal_conv1d 1.7.0，2026-10-12 已装并验证）。

- MambaBlock = 预 norm（默认 RMSNorm，norm_type 可选 LayerNorm——fmt 变体 default-off 钩子）+
  mamba_ssm.Mamba + 残差（因果单向）
- 输入/输出 (B, L, d_model)；past 前缀天然作为条件被 future 位置经递归看到

norm_type default-off 钩子（用户 2026-10-16 拍板 D4：fmt 变体 RMSNorm→LayerNorm）：
默认 "rmsnorm" = nn.RMSNorm 原结构逐位一致（state_dict 键/旧变体 config_hash 零漂移）；
"layernorm" = nn.LayerNorm（含 bias ⇒ state_dict 多 *.norm.bias 键，仅 fmt ckpt）；
先例 bottleneck_rank/early_stop_patience/lr_schedule（getattr(cfg, "norm_type", "rmsnorm") 透传）。
"""

from __future__ import annotations

import torch.nn as nn
from mamba_ssm import Mamba


def build_norm(d_model: int, norm_type: str = "rmsnorm") -> nn.Module:
    """norm 构建钩子：'rmsnorm'（默认，原结构）| 'layernorm'（fmt 变体）；其他值构建期报错。"""
    if norm_type == "rmsnorm":
        return nn.RMSNorm(d_model)
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    raise ValueError(f"unknown norm_type: {norm_type}（可选 'rmsnorm' | 'layernorm'）")


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
        norm_type: str = "rmsnorm",  # default-off 钩子（D4）；默认 = 原 nn.RMSNorm 逐位一致
    ):
        super().__init__()
        self.norm = build_norm(d_model, norm_type)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


def build_mamba_backbone(
    n_layers: int,
    d_model: int = 16,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    use_fast_path: bool = True,
    norm_type: str = "rmsnorm",  # default-off 钩子（D4）：透传各 MambaBlock pre-norm
) -> nn.Sequential:
    return nn.Sequential(
        *[
            MambaBlock(d_model, d_state, d_conv, expand, use_fast_path, norm_type)
            for _ in range(n_layers)
        ]
    )
