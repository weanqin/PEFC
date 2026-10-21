"""速度网络 v_θ(x_t, t, X_past)——**v4bg：v4b d_ch=1 基座 + GNN 先验图头**（design §2）。

结构（任务 10-19-gpfm-mamba-v4b-gnn-f0）：
- Token 序列 512 = past 384（条件前缀）+ future 128（流变量位置），与 v4b 相同；
- 逐通道特征 18 维 = 本通道值 1 + 段类型嵌入 1 + t 正弦嵌入 16；
- [1] 共享 Linear(18→1)（CI 权重跨通道共享）→ (B×62, 512, 1)；
- [2] 4×MambaBlock(1, d_state=16, d_conv=4, expand=2) 逐通道独立 selective scan（复用 v1
  mamba.py 零改动）——**trunk 严格零跨通道混合，与 v4b-d1 逐位同构**（本文件构建顺序 =
  seg → in_proj → blocks → head_norm → head，RNG 消耗顺序与 v4 model.py 一致，同 seed 下
  trunk 参数逐位相同——单变量归因的口径基础，design §0/§2.1）；
- [3] 截取 future 128 + 逐通道 RMSNorm(1)（共享）→ (B, T, C, d_ch)；
- [4] **GraphHead（唯一跨通道混合层**，替换 v4b 的稠密 cMLP `_head`，见 graph_head.py）：
  per-channel Linear(1→h) → K×{A_eff 聚合 + per-channel Linear+GELU} → per-channel
  Linear(h→1)；A_eff = γ_m·M(d;σ) + γ·W（LearnableMexHatMixer 适配，A_eff 构造与源数值等价）。
- 输出 v_θ (B,62,128)；return_all=True 时 head 计算全部 512 位置返回 (B,62,512)
  （与 v1/v4 口径一致，供因果性单测取 past 位置输出）。

运行时契约（design §2.5）：
- **类名沿用 `VelocityNetwork`、按名导出 `EMA`**——v1 train.py/sample.py 按名导入；
- **钩子 kwargs 契约**：现行 v1 train.py:278-280 与 sample.py:52-54 无条件传
  `bottleneck_rank / norm_type / cond_adaln`——本类接受三 kwargs 并**断言默认值**
  （None / "rmsnorm" / "off"；非默认报错——本变体不含这些分支，防隐性口径漂移）；
- graph_* 超参沿 v4 `d_z/head_act` 先例：**类默认 + config↔model 一致性断言**
  （包装脚本运行时校验；禁模块级 `import config`，保 pytest 隔离）；
- 几何加载收敛到 GraphHead 构造**单一 choke point**（train/evaluate/graph_readout 三入口共享）。

`time_embedding` / `EMA` 复制自 v1 model.py，勿单独演化（演化须回填 v1 并重跑回归）。
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from graph_head import (
    DEFAULT_DATA_ROOT,
    DEFAULT_GEOMETRY_PATH,
    GraphHead,
)
from mamba import build_mamba_backbone


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


def _load_w_init(graph_w_init: str) -> Optional[torch.Tensor]:
    """graph_w_init 路径 → (62,62) 张量（.npy/.pt）；空串 = None（零初始化，源默认）。"""
    if graph_w_init == "":
        return None
    if graph_w_init.endswith(".npy"):
        w = torch.from_numpy(np.load(graph_w_init)).float()
    else:
        w = torch.load(graph_w_init, map_location="cpu", weights_only=False).float()
    if w.shape != (62, 62):
        raise ValueError(f"graph_w_init 形状 {tuple(w.shape)} != (62,62)（{graph_w_init}）")
    return w


class VelocityNetwork(nn.Module):
    """v4bg 速度网络（v4b d_ch=1 基座 + GraphHead；类名沿用 v1 以满足按名导入契约）。"""

    def __init__(
        self,
        n_channels: int = 62,
        past_len: int = 384,
        future_len: int = 128,
        d_model: int = 1,          # v4-B 语义 = 逐通道独立宽度 d_ch；本变体固定 1（v4b-d1 臂基座）
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_emb_dim: int = 16,
        use_fast_path: bool = True,
        # ---- GraphHead 超参（config.graph_* 同值，双处一致性防线）----
        graph_kernel: str = "ricker",
        graph_k: int = 2,
        graph_hidden: int = 64,
        graph_gamma: float = 0.1,
        graph_gamma_m: float = 1.0,
        graph_w_init: str = "",     # 空 = W 零初始化；否则 npy/pt 路径注入 (62,62)
        geometry_path: str = DEFAULT_GEOMETRY_PATH,   # 几何单一 choke point（config.geometry_path 同值）
        data_root: str = DEFAULT_DATA_ROOT,           # 通道名 46 例断言用（config.data_root 同值）
        # ---- v1 train.py/sample.py 无条件透传的钩子 kwargs（§2.5 契约：接受并断言默认值）----
        bottleneck_rank: Optional[int] = None,  # 本变体无低秩接口分支——非 None 即口径漂移，报错
        norm_type: str = "rmsnorm",             # 本变体无 layernorm 分支——非 rmsnorm 报错
        cond_adaln: str = "off",                # 本变体无 adaLN 分支——非 off 报错
    ):
        super().__init__()
        if bottleneck_rank is not None:
            raise ValueError(
                f"v4bg 不支持 bottleneck_rank（本变体无低秩输入接口；got {bottleneck_rank}）——"
                "口径漂移防护：如需启用另立变体并登记"
            )
        if norm_type != "rmsnorm":
            raise ValueError(
                f"v4bg 不支持 norm_type={norm_type!r}（本变体 norm 固定 rmsnorm；fmt 系口径另立变体）"
            )
        if cond_adaln != "off":
            raise ValueError(
                f"v4bg 不支持 cond_adaln={cond_adaln!r}（本变体无 adaLN 分支；adaln 口径另立变体）"
            )
        if d_model < 1:
            raise ValueError(f"d_ch(=d_model) 须 ≥1，got {d_model}")
        self.n_channels = n_channels
        self.past_len = past_len
        self.future_len = future_len
        self.L = past_len + future_len
        self.time_emb_dim = time_emb_dim
        self.d_ch = d_model        # 别名登记：d_model 即 d_ch（v4-B 惯例）
        self.graph_kernel = graph_kernel
        self.graph_k = graph_k
        self.graph_hidden = graph_hidden
        self.graph_gamma = graph_gamma
        self.graph_gamma_m = graph_gamma_m
        self.seg_past = nn.Parameter(torch.zeros(()))   # 段类型嵌入：可学习标量（不消耗 RNG）
        self.seg_future = nn.Parameter(torch.zeros(()))
        # [1] 逐通道独立输入投影（权重跨通道共享）：18 = 1(本通道值)+1(seg)+16(t_emb)
        self.in_proj = nn.Linear(1 + 1 + time_emb_dim, d_model)
        # [2] CI 骨干（权重跨通道共享，通道折进 batch 维；RNG 消耗顺序与 v4b 一致）
        self.blocks = build_mamba_backbone(
            n_layers, d_model, d_state, d_conv, expand, use_fast_path
        )
        # [3] head 前逐通道 RMSNorm(d_ch)（共享；不消耗 RNG）
        self.head_norm = nn.RMSNorm(d_model)
        # [4] GraphHead（唯一跨通道混合层）：几何 choke point 在其构造内（不消耗 RNG）
        self.graph_head = GraphHead(
            n_channels=n_channels,
            d_ch=d_model,
            hidden=graph_hidden,
            k_layers=graph_k,
            gamma=graph_gamma,
            gamma_m=graph_gamma_m,
            kernel=graph_kernel,
            w_init=_load_w_init(graph_w_init),
            geometry_path=geometry_path,
            data_root=data_root,
        )

    # ---------------- CI trunk（通道 j 输入只影响通道 j 的 h） ----------------

    def trunk_forward(
        self,
        x_past: torch.Tensor,  # (B, C, P)
        x_f: torch.Tensor,     # (B, C, F)
        t: torch.Tensor,       # (B,)
    ) -> torch.Tensor:         # (B, C, L, d_ch)
        """CI 编码 + Mamba 骨干：EEG 通道间零信息流动（与 v4b 逐位同构）。"""
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

    # ---------------- 图头读出（唯一跨通道混合层） ----------------

    def _pool(self, h_pos: torch.Tensor) -> torch.Tensor:
        """逐通道 RMSNorm + future 布局整理：(B, C, T, d_ch) → (B, T, C, d_ch)（GraphHead 接口）。"""
        B, Cn, T, d_ch = h_pos.shape
        hn = self.head_norm(h_pos.reshape(B * Cn, T, d_ch)).reshape(B, Cn, T, d_ch)
        return hn.permute(0, 2, 1, 3)                     # (B, T, C, d_ch)

    def forward(
        self,
        x_past: torch.Tensor,  # (B, C, P)
        x_f: torch.Tensor,     # (B, C, F) 流变量当前值
        t: torch.Tensor,       # (B,)
        return_all: bool = False,
    ) -> torch.Tensor:
        h = self.trunk_forward(x_past, x_f, t)             # (B, C, L, d_ch)
        if return_all:
            out = self.graph_head(self._pool(h))           # (B, L, C) head 过全部位置
        else:
            out = self.graph_head(self._pool(h[:, :, self.past_len :]))  # (B, F, C) 截取 future
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
