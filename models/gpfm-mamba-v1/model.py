"""速度网络 v_θ(x_t, t, X_past)（design §3）。

- Token 序列 512：前 384 = past 条件前缀，后 128 = future 流变量位置
- 逐时间步输入 = 62 通道值 + 段类型嵌入（past/future 可学习标量）+（future 位置）全局 t 的正弦嵌入（dim 16 广播）
- 投影 Linear(→16) → 4×MambaBlock(d_model=16, d_state=16, d_conv=4, expand=2) → 输出头
  norm + Linear(16→62)（norm 默认 RMSNorm，norm_type="layernorm" 时为 LayerNorm——fmt 变体
  D4 default-off 钩子，5 个 norm 位 = 4×MambaBlock pre-norm + 本 head norm）
- cond_adaln default-off 钩子（任务 10-17-gpfm-mamba-adaln-f0，A 案 scale/shift）：默认 "off"
  分支与原结构逐位一致（不建新参数/子模块）；"scale_shift" = 5 个 norm 位全部换
  `modulate(LN(x), s(t), b(t)) = (1+s)⊙LN(x)+b`，(s,b) 由 time_embedding(t,16) 经逐 site
  Sequential(SiLU, Linear(16, 2*d_model))（fork_rng 内构建 + 末层零初始化 ⇒ 初始等价纯 LN）
  生成——块位键 blocks.{i}.mod.1.*、head 位键 head_mod.1.*；forward 签名不变（t 已在参里）
- 仅取 future 128 位置 → v_θ (B,62,128)（return_all=True 供因果性单测取全部位置）
- EMA 封装（decay 由 config 传入，默认 0.999），推理/评估用 EMA 权重
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn

from mamba import build_mamba_backbone, build_norm, modulate


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
        bottleneck_rank: int | None = None,  # v4a 变体启用（低秩输入瓶颈）；None = 原结构逐位不变
        norm_type: str = "rmsnorm",  # fmt 变体启用（D4 RMSNorm→LayerNorm）；默认 = 原结构逐位不变
        cond_adaln: str = "off",     # adaln 变体启用（A 案 scale/shift）；默认 = 原结构逐位不变
    ):
        super().__init__()
        self.n_channels = n_channels
        self.past_len = past_len
        self.future_len = future_len
        self.L = past_len + future_len
        self.time_emb_dim = time_emb_dim
        self.cond_adaln = cond_adaln  # 模式字符串属性（非参数/子模块——"off" 分支结构零漂移）
        self.seg_past = nn.Parameter(torch.zeros(()))   # 段类型嵌入：可学习标量
        self.seg_future = nn.Parameter(torch.zeros(()))
        # 低秩输入瓶颈 default-off 钩子（v4-A，用户 2026-10-16 拍板；先例 prior_bandlimit_hz/
        # early_stop_patience/lr_schedule）：None 分支与原实现逐位一致（结构/state_dict 键不变，
        # v1/v2/v3 行为零漂移）；非 None 分支 = 纯线性两层 Sequential(Linear(in→r), Linear(r→d_model))，
        # 无激活（两层复合 = rank≤r 线性映射——接口强制跨通道混合的干净单变量）、PyTorch 默认初始化，
        # state_dict 键 in_proj.{0,1}.{weight,bias}。forward 不变（Linear 与 Sequential 同可调用）。
        if bottleneck_rank is None:
            self.in_proj = nn.Linear(n_channels + 1 + time_emb_dim, d_model)
        else:
            self.in_proj = nn.Sequential(
                nn.Linear(n_channels + 1 + time_emb_dim, bottleneck_rank),
                nn.Linear(bottleneck_rank, d_model),
            )
        # norm 类型 default-off 钩子（用户 2026-10-16 拍板 D4，fmt 变体 layernorm；先例
        # bottleneck_rank）：默认 "rmsnorm" 分支与原实现逐位一致（结构/state_dict 键不变，
        # v1/v2/v3/v4a 行为零漂移）；"layernorm" = 5 个 norm 位（4×MambaBlock pre-norm +
        # head norm）全部 nn.LayerNorm（含 bias ⇒ state_dict 多 *.norm.bias/head_norm.bias 键）。
        self.blocks = build_mamba_backbone(
            n_layers, d_model, d_state, d_conv, expand, use_fast_path, norm_type,
            cond_adaln=cond_adaln, mod_dim=time_emb_dim,  # adaln default-off 钩子透传（off 时 mod_dim 不使用）
        )
        self.head_norm = build_norm(d_model, norm_type)
        self.head = nn.Linear(d_model, n_channels)
        # adaLN head 位 default-off 钩子（A 案）：启用时（同样 fork_rng 内——RNG 中性构造契约）建
        # head 位调制 MLP（零初始化 ⇒ 初始严格等价纯 LayerNorm head）；off 时不建任何新参数/子模块。
        if cond_adaln == "scale_shift":
            with torch.random.fork_rng(devices=[]):
                self.head_mod = nn.Sequential(
                    nn.SiLU(), nn.Linear(time_emb_dim, 2 * d_model)
                )
                nn.init.zeros_(self.head_mod[-1].weight)
                nn.init.zeros_(self.head_mod[-1].bias)

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
        if self.cond_adaln == "scale_shift":
            # adaLN 分支（A 案）：调制侧条件 = time_embedding(t) 再算一次 (B, time_emb_dim)，
            # 与输入侧拼接（仅 future 位）并存且不对称——DiT 同款语义（timestep 既拼 token 也驱动 adaLN），
            # per-sample 全局调制广播全 512 位置（含 past 段；Euler 每步全前向重算，采样自洽）。
            tcond = time_embedding(t, self.time_emb_dim)
            for blk in self.blocks:  # Sequential 不能直传额外参 ⇒ 显式块循环（仅本分支）
                h = blk(h, tcond)
            s_h, b_h = self.head_mod(tcond).chunk(2, dim=-1)  # (B,2d) -> s/b 各 (B,d)
            out = self.head(modulate(self.head_norm(h), s_h, b_h))  # (B, L, C)
        else:
            h = self.blocks(h)
            out = self.head(self.head_norm(h))  # (B, L, C) 位置无关头
        if return_all:
            return out.permute(0, 2, 1)     # (B, C, L)
        return out[:, self.past_len :].permute(0, 2, 1)  # v_θ (B, C, F)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EMA:
    """指数滑动平均权重（decay 由 config 传入，默认 0.999——短程训练口径）；评估经 swap 上下文使用 EMA 权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
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
