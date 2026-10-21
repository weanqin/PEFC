"""速度网络 v_θ(x_t, t, X_past) — gpfm-mamba-v5（v2 基线 × 轴分解交替，design §2.1）。

结构（用户 2026-10-19 四项要求的逐字实现）：
- 输入构造与 in_proj/head 与 v2 逐字一致（复制自 v1 model.py，含 time_embedding/EMA）：
  seq+seg+temb → (B,512,79) → in_proj Linear(79→62) → ... → head_norm RMSNorm(62) +
  head Linear(62→62) → 截取 future 128 → v_θ (B,62,128)（return_all=True 全 512）。
- **trunk = 4×[Mamba 时间块 → GNN 空间块 → LayerNorm] 逐块交替**（顺序严格
  Mamba → 空间块 → LN，用户指定；不能整段 Sequential——空间块需 A_eff 形参）：
    时间块 = MambaBlock(62)：x + Mamba(RMSNorm(x))（v1 mamba.py 只读复用，时间轴 L）；
    空间块 = h + GELU(msg_i(A_eff @_axis h))（graph.py；A_eff 跨块共享单实例）；
    LayerNorm = nn.LayerNorm(62)（逐块独立实例，post-norm，无残差包裹——用户指定，
    不走 norm_type 钩子）。
- **d_model = C = 62**：模型维即通道轴，零折叠/合并改变对应（构造断言 d_model==n_channels，
  不等即报错——要求③的机器可读防线）。
- 钩子三参（bottleneck_rank/norm_type/cond_adaln）**接受并断言默认值**
  （None/"rmsnorm"/"off"，非默认报错——B-2 处置沿用；v1 train.py/sample.py 无条件透传
  可直跑，默认分支行为与断言一致）。
- graph_* 新参类默认 == config 字段（v1 train/sample 构建不透传 graph_*；双遮蔽包装的
  运行时一致性断言防线同 v4 惯例）。
- RNG 注记：v5 为固定结构（无 default-off 分支），构建期仅 4×msg Linear 消耗全局 RNG，
  顺序固定（blocks → graph → spatial → lns），无 fork_rng 契约需求。
- EMA 封装复制自 v1（勿单独演化惯例），推理/评估用 EMA 权重。
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator

import numpy as np
import torch
import torch.nn as nn

from graph import DEFAULT_GEOMETRY_PATH, GraphSpatialStep, MexHatAdjacency
from mamba import build_mamba_backbone, build_norm


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """标量 t∈[0,1] 的正弦嵌入：sin/cos 各 dim/2 个频率（1→1e-4，×1000 尺度）。（v1 逐字复制）"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.reshape(-1, 1).float() * 1000.0 * freqs.reshape(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


def _resolve_w_init(graph_w_init):
    """config 的 graph_w_init 字符串 → W 初始张量（或 None=零初始化）。

    ""：零初始化（默认）；str：(.pt/.npy) 文件路径装载 (62,62) 有向拓扑；
    torch.Tensor：直接克隆（程序化/单测注入用）。
    """
    if graph_w_init is None or graph_w_init == "":
        return None
    if isinstance(graph_w_init, torch.Tensor):
        return graph_w_init.clone().float()
    p = str(graph_w_init)
    if p.endswith(".pt"):
        return torch.as_tensor(torch.load(p, map_location="cpu", weights_only=False)).float()
    if p.endswith(".npy"):
        return torch.as_tensor(np.load(p)).float()
    raise ValueError(f"graph_w_init 仅支持 ''（零初始化）/.pt/.npy 路径或张量，got: {graph_w_init!r}")


class VelocityNetwork(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        past_len: int = 384,
        future_len: int = 128,
        d_model: int = 62,          # v5：模型维即通道轴（=62；断言 d_model==n_channels）
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_emb_dim: int = 16,
        use_fast_path: bool = True,
        bottleneck_rank: int | None = None,  # 钩子（v4a）：v5 结构不含瓶颈分支——仅接受默认 None
        norm_type: str = "rmsnorm",          # 钩子（D4）：v5 不换 norm——仅接受默认 "rmsnorm"
        cond_adaln: str = "off",             # 钩子（A 案）：v5 无 adaLN——仅接受默认 "off"
        # ---- v5 图空间块新参（类默认 == config 字段，运行时断言双处一致）----
        graph_kernel: str = "ricker",
        graph_gamma: float = 0.1,
        graph_gamma_m: float = 1.0,
        graph_w_init: str | torch.Tensor = "",   # "" = W 零初始化；路径/张量 = 注入（见 _resolve_w_init）
        geometry_path: str = DEFAULT_GEOMETRY_PATH,
    ):
        super().__init__()
        # 钩子三参断言默认（B-2 处置沿用：非默认值构建期报错，防 config 与结构静默漂移）。
        if bottleneck_rank is not None:
            raise ValueError(f"v5 无瓶颈分支，bottleneck_rank 仅接受 None，got: {bottleneck_rank}")
        if norm_type != "rmsnorm":
            raise ValueError(f"v5 保持 RMSNorm（MambaBlock 内 pre-norm 与 head_norm），norm_type 仅接受 'rmsnorm'，got: {norm_type!r}")
        if cond_adaln != "off":
            raise ValueError(f"v5 无 adaLN 分支，cond_adaln 仅接受 'off'，got: {cond_adaln!r}")
        # 要求③机器可读防线：模型维即通道轴（零折叠/合并）。
        if d_model != n_channels:
            raise ValueError(
                f"v5 轴分解不变量破坏：d_model ({d_model}) 必须等于 n_channels ({n_channels})"
                "（d_model=C=62——通道不折叠进 batch、不以投影改变对应）"
            )
        self.n_channels = n_channels
        self.past_len = past_len
        self.future_len = future_len
        self.L = past_len + future_len
        self.time_emb_dim = time_emb_dim
        self.n_layers = n_layers
        self.seg_past = nn.Parameter(torch.zeros(()))   # 段类型嵌入：可学习标量（v2 逐字）
        self.seg_future = nn.Parameter(torch.zeros(()))
        self.in_proj = nn.Linear(n_channels + 1 + time_emb_dim, d_model)  # (B,L,79)→(B,L,62)（v2 逐字）
        # 时间块骨干（v1 mamba.py 只读复用；构建后**逐块迭代**——空间块须插在每块之后，
        # 不能整段 Sequential）。state_dict 键 blocks.{i}.* 与 v2 同构。
        self.blocks = build_mamba_backbone(
            n_layers, d_model, d_state, d_conv, expand, use_fast_path, norm_type,
            cond_adaln=cond_adaln, mod_dim=time_emb_dim,  # off 时不建调制参数（已断言）
        )
        # 空间块共享 A_eff 单实例（唯一可解释连接矩阵）+ 每块独立 msg_i / LN_i。
        self.graph = MexHatAdjacency(
            n_channels=n_channels, coords=None, kernel=graph_kernel,
            gamma=graph_gamma, gamma_m=graph_gamma_m,
            w_init=_resolve_w_init(graph_w_init), geometry_path=geometry_path,
        )
        self.spatial_blocks = nn.ModuleList(
            [GraphSpatialStep(d_model) for _ in range(n_layers)]
        )
        self.block_lns = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
        self.head_norm = build_norm(d_model, norm_type)   # RMSNorm(62)（v2 逐字）
        self.head = nn.Linear(d_model, n_channels)        # (B,L,62)→(B,L,62)（v2 逐字）

    def forward(
        self,
        x_past: torch.Tensor,  # (B, C, P)
        x_f: torch.Tensor,     # (B, C, F) 流变量当前值
        t: torch.Tensor,       # (B,)
        return_all: bool = False,
    ) -> torch.Tensor:
        B = x_past.shape[0]
        # ---- 输入构造（v2 逐字：seq + seg + temb → in_proj）----
        seq = torch.cat([x_past, x_f], dim=-1).permute(0, 2, 1)  # (B, L, C)
        seg = seq.new_empty(B, self.L, 1)
        seg[:, : self.past_len] = self.seg_past
        seg[:, self.past_len :] = self.seg_future
        temb = seq.new_zeros(B, self.L, self.time_emb_dim)
        temb[:, self.past_len :] = time_embedding(t, self.time_emb_dim).unsqueeze(1)
        h = self.in_proj(torch.cat([seq, seg, temb], dim=-1))  # (B, L, d_model=62)
        # ---- 交替主体：4×[Mamba 时间块 → 空间块 → LayerNorm]（逐块，顺序严格）----
        A_eff = self.graph.effective_adjacency()  # (C, C) 跨块共享单实例，逐块同一矩阵
        for i in range(self.n_layers):
            h = self.blocks[i](h)                 # 时间块：x + Mamba(RMSNorm(x))，时间轴 L
            h = self.spatial_blocks[i](h, A_eff)  # 空间块：h + GELU(msg_i(A_eff @_axis h))
            h = self.block_lns[i](h)              # post-LayerNorm(62)（无残差包裹）
        # ---- 输出头（v2 逐字）----
        out = self.head(self.head_norm(h))        # (B, L, C) 位置无关头
        if return_all:
            return out.permute(0, 2, 1)     # (B, C, L)
        return out[:, self.past_len :].permute(0, 2, 1)  # v_θ (B, C, F)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EMA:
    """指数滑动平均权重（decay 由 config 传入，默认 0.999——短程训练口径）；评估经 swap 上下文使用 EMA 权重。（v1 逐字复制）"""

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
