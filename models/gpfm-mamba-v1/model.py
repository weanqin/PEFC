"""速度网络 v_θ(x_t, t, X_past)（design §3）。

- Token 序列 512：前 384 = past 条件前缀，后 128 = future 流变量位置
- 逐时间步输入 = 62 通道值 + 段类型嵌入（past/future 可学习标量）+（future 位置）全局 t 的正弦嵌入（dim 16 广播）
- 投影 Linear(→16) → 4×MambaBlock(d_model=16, d_state=16, d_conv=4, expand=2) → 输出头 RMSNorm+Linear(16→62)
- 仅取 future 128 位置 → v_θ (B,62,128)（return_all=True 供因果性单测取全部位置）
- EMA（decay 0.9999）封装，推理/评估用 EMA 权重
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn

from mamba import build_mamba_backbone


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """标量 t∈[0,1] 的正弦嵌入：sin/cos 各 dim/2 个频率（1→1e-4，×1000 尺度）。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.reshape(-1, 1).float() * 1000.0 * freqs.reshape(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class VelocityNetwork(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        past_len: int = 384,
        future_len: int = 128,
        d_model: int = 16,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_emb_dim: int = 16,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.past_len = past_len
        self.future_len = future_len
        self.L = past_len + future_len
        self.time_emb_dim = time_emb_dim
        self.seg_past = nn.Parameter(torch.zeros(()))   # 段类型嵌入：可学习标量
        self.seg_future = nn.Parameter(torch.zeros(()))
        self.in_proj = nn.Linear(n_channels + 1 + time_emb_dim, d_model)
        self.blocks = build_mamba_backbone(
            n_layers, d_model, d_state, d_conv, expand, use_fast_path
        )
        self.head_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, n_channels)

    def forward(
        self,
        x_past: torch.Tensor,  # (B, C, P)
        x_f: torch.Tensor,     # (B, C, F) 流变量当前值
        t: torch.Tensor,       # (B,)
        return_all: bool = False,
    ) -> torch.Tensor:
        B = x_past.shape[0]
        seq = torch.cat([x_past, x_f], dim=-1).permute(0, 2, 1)  # (B, L, C)
        seg = seq.new_empty(B, self.L, 1)
        seg[:, : self.past_len] = self.seg_past
        seg[:, self.past_len :] = self.seg_future
        temb = seq.new_zeros(B, self.L, self.time_emb_dim)
        temb[:, self.past_len :] = time_embedding(t, self.time_emb_dim).unsqueeze(1)
        h = self.in_proj(torch.cat([seq, seg, temb], dim=-1))  # (B, L, d_model)
        h = self.blocks(h)
        out = self.head(self.head_norm(h))  # (B, L, C) 位置无关头
        if return_all:
            return out.permute(0, 2, 1)     # (B, C, L)
        return out[:, self.past_len :].permute(0, 2, 1)  # v_θ (B, C, F)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EMA:
    """指数滑动平均权重（论文口径 decay=0.9999）；评估经 swap 上下文使用 EMA 权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().to(s.device), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def to(self, device) -> "EMA":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: Dict) -> None:
        self.decay = sd["decay"]
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}

    @contextmanager
    def swap(self, model: nn.Module) -> Iterator[None]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(
            {k: v.to(dtype=backup[k].dtype, device=backup[k].device) for k, v in self.shadow.items()}
        )
        try:
            yield
        finally:
            model.load_state_dict(backup)
