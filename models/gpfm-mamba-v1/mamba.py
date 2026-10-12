"""官方 mamba-ssm 适配层（mamba_ssm 2.3.1 + causal_conv1d 1.7.0，2026-10-12 已装并验证）。

- MambaBlock = 预 RMSNorm + mamba_ssm.Mamba + 残差（因果单向）
- 输入/输出 (B, L, d_model)；past 前缀天然作为条件被 future 位置经递归看到
"""

from __future__ import annotations

import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
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
) -> nn.Sequential:
    return nn.Sequential(
        *[
            MambaBlock(d_model, d_state, d_conv, expand, use_fast_path)
            for _ in range(n_layers)
        ]
    )
