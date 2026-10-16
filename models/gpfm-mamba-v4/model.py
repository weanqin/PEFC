"""速度网络 v_θ(x_t, t, X_past)——**v4-B：通道独立骨干 + cMLP 组件式读出**（design §2）。

用户 2026-10-16 指定架构（别名：口头「v3」「v4-B」均指本变体）：
- Token 序列 512 = past 384（条件前缀）+ future 128（流变量位置），与 v1/v2 相同；
- 逐通道特征 18 维 = 本通道值 1 + 段类型嵌入 1 + t 正弦嵌入 16（t_emb 仅 future 位置非零，
  广播到全部通道——与 v1 同源）；
- [1] 共享 Linear(18→d_ch)（CI 权重跨通道共享）→ (B×62, 512, d_ch)；
- [2] 4×MambaBlock(d_ch, d_state=16, d_conv=4, expand=2) 逐通道独立 selective scan
  （复用 v1 mamba.py 零改动；块结构 = 预 RMSNorm + Mamba + 残差）——**trunk 严格零跨通道混合**；
- [3] 截取 future 128 + 逐通道 RMSNorm(d_ch)（共享）→ 同时间步汇聚 h_t∈R^{62·d_ch}；
- [4] cMLP head（**唯一跨通道混合层**，62 个独立子 MLP）：z_{t,i}=σ(W^{(i),1}h_t+b¹_i)、
  v_θ(t,i)=W^{(i),2}z_{t,i}+b²_i；W¹ 按源通道列分块 W^{(i),1}=[W_{:,1}|…|W_{:,62}]
  （块形状 (d_z, d_ch)）——堆叠参数张量 + einsum 实现，无逐通道 Python 循环。
- 输出 v_θ (B,62,128)；return_all=True 时 head 计算全部 512 位置返回 (B,62,512)
  （与 v1 口径一致，供因果性单测取 past 位置输出）。

运行时契约（design §4.3，v1 train.py/sample.py 按名导入）：
- **类名沿用 `VelocityNetwork`、按名导出 `EMA`**——构造签名与 v1 完全一致
  （n_channels…use_fast_path 全部接受；`d_model` 在 v4-B 中的语义 = 逐通道宽度 d_ch），
  额外类默认参数 d_z=16 / head_act="silu"（与 config 字段同值，双处一致性防线）；
- `forward(x_past, x_f, t, return_all=False)` 输入输出形状与 v1 相同；`num_params()`。
- 初始化全部 PyTorch 默认（head 等价于 62 个堆叠 nn.Linear 的默认 kaiming-uniform/uniform-bias）。

`time_embedding` / `EMA` 复制自 v1 model.py，勿单独演化（演化须回填 v1 并重跑回归）。
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import build_mamba_backbone

_HEAD_ACTS = {"silu": F.silu}


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """标量 t∈[0,1] 的正弦嵌入：sin/cos 各 dim/2 个频率（1→1e-4，×1000 尺度）。

    （复制自 v1 model.py，勿单独演化。）
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.reshape(-1, 1).float() * 1000.0 * freqs.reshape(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class VelocityNetwork(nn.Module):
    """v4-B 速度网络（CI 骨干 + cMLP head；类名沿用 v1 以满足按名导入契约）。"""

    def __init__(
        self,
        n_channels: int = 62,
        past_len: int = 384,
        future_len: int = 128,
        d_model: int = 8,          # v4-B 语义 = 逐通道独立宽度 d_ch（config.d_model=8 同值；
                                   # 口径变更 2026-10-16 方案 B：初值 1 → 8，config+model 双处同改）
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_emb_dim: int = 16,
        use_fast_path: bool = True,
        d_z: int = 16,             # v4-B 特有：cMLP head 子 MLP 隐宽（config.d_z 同值）
        head_act: str = "silu",    # v4-B 特有：head 激活 σ（config.head_act 同值）
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_ch(=d_model) 须 ≥1，got {d_model}")
        if head_act not in _HEAD_ACTS:
            raise ValueError(f"未知 head_act: {head_act}（可选 {sorted(_HEAD_ACTS)}）")
        self.n_channels = n_channels
        self.past_len = past_len
        self.future_len = future_len
        self.L = past_len + future_len
        self.time_emb_dim = time_emb_dim
        self.d_ch = d_model        # 别名登记：v4-B 中 d_model 即 d_ch
        self.d_z = d_z
        self.head_act = head_act
        self.act = _HEAD_ACTS[head_act]
        self.seg_past = nn.Parameter(torch.zeros(()))   # 段类型嵌入：可学习标量
        self.seg_future = nn.Parameter(torch.zeros(()))
        # [1] 逐通道独立输入投影（权重跨通道共享）：18 = 1(本通道值)+1(seg)+16(t_emb)
        self.in_proj = nn.Linear(1 + 1 + time_emb_dim, d_model)
        # [2] CI 骨干（权重跨通道共享，通道折进 batch 维）
        self.blocks = build_mamba_backbone(
            n_layers, d_model, d_state, d_conv, expand, use_fast_path
        )
        # [3] head 前逐通道 RMSNorm(d_ch)（共享）
        self.head_norm = nn.RMSNorm(d_model)
        # [4] cMLP head：W¹ (62, d_z, 62·d_ch) 第 i 片 = 目标通道 i 的第一层权重，
        #     最后维按源通道 j 列分块 [j·d_ch:(j+1)·d_ch)；初始化 = 62 个堆叠
        #     nn.Linear 的 PyTorch 默认（kaiming_uniform(a=√5) + uniform bias 1/√fan_in）。
        k_head = n_channels * d_model
        w1 = torch.empty(n_channels, d_z, k_head)
        nn.init.kaiming_uniform_(w1.view(n_channels * d_z, k_head), a=math.sqrt(5))
        self.W1 = nn.Parameter(w1)
        b1 = torch.empty(n_channels, d_z)
        bound1 = 1.0 / math.sqrt(k_head)
        nn.init.uniform_(b1, -bound1, bound1)
        self.b1 = nn.Parameter(b1)
        w2 = torch.empty(n_channels, d_z)               # 第 i 行 = 目标 i 的第二层权重 (d_z,)
        nn.init.kaiming_uniform_(w2, a=math.sqrt(5))
        self.W2 = nn.Parameter(w2)
        b2 = torch.empty(n_channels)
        bound2 = 1.0 / math.sqrt(d_z)
        nn.init.uniform_(b2, -bound2, bound2)
        self.b2 = nn.Parameter(b2)

    # ---------------- CI trunk（通道 j 输入只影响通道 j 的 h） ----------------

    def trunk_forward(
        self,
        x_past: torch.Tensor,  # (B, C, P)
        x_f: torch.Tensor,     # (B, C, F)
        t: torch.Tensor,       # (B,)
    ) -> torch.Tensor:         # (B, C, L, d_ch)
        """CI 编码 + Mamba 骨干：EEG 通道间零信息流动（CI 隔离性单测口径）。"""
        B = x_past.shape[0]
        Cn = self.n_channels
        seq = torch.cat([x_past, x_f], dim=-1).permute(0, 2, 1)  # (B, L, C)
        seg = seq.new_empty(B, self.L, 1)
        seg[:, : self.past_len] = self.seg_past
        seg[:, self.past_len :] = self.seg_future
        temb = seq.new_zeros(B, self.L, self.time_emb_dim)
        temb[:, self.past_len :] = time_embedding(t, self.time_emb_dim).unsqueeze(1)
        feat = torch.cat(
            [seq.unsqueeze(-1),                          # (B, L, C, 1) 本通道值
             seg.unsqueeze(2).expand(B, self.L, Cn, 1),  # (B, L, C, 1) 段类型（共享标量广播）
             temb.unsqueeze(2).expand(B, self.L, Cn, self.time_emb_dim)],  # 全局 t_emb 广播
            dim=-1,
        )                                                  # (B, L, C, 18)
        h = self.in_proj(feat)                            # (B, L, C, d_ch) 共享投影
        h = h.permute(0, 2, 1, 3).reshape(B * Cn, self.L, self.d_ch)  # 通道折进 batch
        h = self.blocks(h)                                # (B·C, L, d_ch) CI selective scan
        return h.reshape(B, Cn, self.L, self.d_ch)

    # ---------------- cMLP readout（唯一跨通道混合层） ----------------

    def _pool(self, h_pos: torch.Tensor) -> torch.Tensor:
        """逐通道 RMSNorm + 同时间步汇聚：(B, C, T, d_ch) → (B, T, C·d_ch)，列块 j = 通道 j。"""
        B, Cn, T, d_ch = h_pos.shape
        hn = self.head_norm(h_pos.reshape(B * Cn, T, d_ch)).reshape(B, Cn, T, d_ch)
        return hn.permute(0, 2, 1, 3).reshape(B, T, Cn * d_ch)

    def _head(self, g: torch.Tensor) -> torch.Tensor:
        """62 个独立子 MLP 的堆叠 einsum：g (B, T, C·d_ch) → v (B, T, C)。

        z_{b,t,i,:} = σ(Σ_k W1[i,:,k]·g[b,t,k] + b1[i])；v_{b,t,i} = W2[i,:]·z + b2[i]。
        """
        z = self.act(torch.einsum("btk,izk->btiz", g, self.W1) + self.b1)  # (B, T, C, d_z)
        return torch.einsum("btiz,iz->bti", z, self.W2) + self.b2          # (B, T, C)

    def forward(
        self,
        x_past: torch.Tensor,  # (B, C, P)
        x_f: torch.Tensor,     # (B, C, F) 流变量当前值
        t: torch.Tensor,       # (B,)
        return_all: bool = False,
    ) -> torch.Tensor:
        h = self.trunk_forward(x_past, x_f, t)             # (B, C, L, d_ch)
        if return_all:
            out = self._head(self._pool(h))                # (B, L, C) head 过全部位置
        else:
            out = self._head(self._pool(h[:, :, self.past_len :]))  # (B, F, C) 截取 future
        return out.permute(0, 2, 1)                        # (B, C, T)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EMA:
    """指数滑动平均权重（decay 由 config 传入，默认 0.999——短程训练口径）；评估经 swap 上下文使用 EMA 权重。

    （复制自 v1 model.py，勿单独演化。）
    """

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
