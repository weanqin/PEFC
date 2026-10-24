#!/usr/bin/env python3
"""flow_exit_gnn.py — flow 出口 GNN 图混合变体子类（任务 10-24-latent-flow-exit-gnn）.

包体零改动纪律（prd R4）：本模块是变体层子类，latent_flow/ 包体一律不触。

变体内容（design §1.1 唯一结构性修改，tail 段替换）：
  - 前段与包内 FlowVelocityNet._forward_graph 现行逻辑逐行一致（时间嵌入、
    入口 LearnableMexHatMixer 图混合、通道折叠进 batch、6 膨胀卷积块、
    out_norm——复刻 oc_latent_flow.py:649-677）；
  - 尾段替换：out_norm 之后先做 K 轮出口图消息传递——与入口混合器
    LearnableMexHatMixer.forward（oc_latent_flow.py:503-516）同式：
    agg = X @ A_eff^T（通道轴矩阵乘）→ x += GELU(Linear(agg)) 逐轮残差——
    再经零初始化逐通道投影 exit_out(hidden→latent_dim) 输出 v。
  - 与入口的两点差异均有明确理由（design §1.2）：①无 in_proj/t_proj——出口
    输入已是 hidden 且 t 已在入口注入；②无 cat/z_obs 条件——出口处理的是
    flow 内部表示，无外部条件输入。

关键设计事实：
  - **A_eff 与入口共享**（design §2 / 用户 2026-10-24 拍板，弃独立备选）：
    出口消息传递直接取 self.graph_mixer.effective_adjacency()，σ/W/γ_M/γ_W
    单份，入口/出口两路损失的梯度在共享参数上双路累计；stage2 现行
    anti-uniform λ=0.1 自然同时约束这一份 A_eff。
  - **exit_out weight/bias 零初始化 ⇒ 初始 v ≡ 0**（恒等起步）：单步公式
    z_fc_hat = z_t + (1-t)·v 退化为 z_fc_hat = z_t，与原架构 graph_out
    零初始化门控同机制（代码库内已验证的模式）；出口消息层首步被零投影
    门控，梯度先打开 exit_out 再传导至消息层。
  - **del self.graph_out**：防死参数/死键——state_dict 不残留原出口投影，
    架构即凭据，装载侧（viz/quality）以 --flow-exit-gnn 旗标 strict load
    匹配；旗标漏传 = strict load 失败（安全失败模式）。
  - 参数净增 +33,024（design §2）：exit_msg_layers 2×Linear(128,128)
    = 33,024；exit_out 8,256 替换 graph_out 8,256（净 0，原键删除不残留）；
    A_eff 共享 +0 ⇒ flow 488,897 → 521,921。
  - 仅 mexhat/dog 流有效（graph_mixer 须具备 effective_adjacency；gaussian
    分支 GraphChannelMixer 无此方法，dense 分支无 graph_mixer）——构造时
    断言 graph_mode 且 kernel ∈ {mexhat, dog}，否则拒建（断言即拒跑惯例）。

直接运行本模块 = CPU 快速构建自验（内存内随机张量，不读数据不训练）：
  python3 models/latent-flow-v1/flow_exit_gnn.py
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入

from latent_flow.models.oc_latent_flow import FlowVelocityNet  # noqa: E402


# =============================================================================
# 子类：构造替换（del graph_out + exit 模块）+ _forward_graph 尾段覆写
# =============================================================================

class FlowVelocityNetExitGNN(FlowVelocityNet):
    """flow 出口 GNN 图混合变体（design §1.3 实施形态：子类，包体零改动）。

    构造 = super().__init__(同参) 后 del self.graph_out，新建 exit_msg_layers
    （K×Linear(hidden,hidden)，默认 PyTorch 初始化，同入口 msg_layers 构造
    同式）与 exit_out（Linear(hidden,latent)，weight/bias 零初始化）；
    前向 = 覆写 _forward_graph（前段复用 super 构建的 graph_mixer/blocks/
    out_norm，尾段替换为出口图混合块）。
    """

    def __init__(self, n_channels: int = 128, latent_dim: int = 64,
                 t_lat: int = 16, hidden_ch: int = 128,
                 dilations=(1, 2, 4, 8, 16, 32),
                 graph_mode: bool = True, graph_k: int = 2,
                 a_hat: torch.Tensor = None,
                 graph_out_mix_alpha: float = 0.5,
                 graph_kernel: str = "mexhat",
                 coords: torch.Tensor = None,
                 mexhat_gamma: float = 0.1,
                 w_init: torch.Tensor = None,
                 mexhat_gamma_m: float = 1.0,
                 dog_init: dict = None):
        # 构造守卫：出口消息传递依赖 graph_mixer.effective_adjacency()，
        # 仅 mexhat/dog 分支具备——不满足即拒建（断言即拒跑惯例）
        assert graph_mode, \
            "FlowVelocityNetExitGNN 仅支持 graph 流（graph_mode=True）"
        assert graph_kernel in ("mexhat", "dog"), (
            f"FlowVelocityNetExitGNN 仅支持 mexhat/dog 流（graph_mixer 须有 "
            f"effective_adjacency），got kernel={graph_kernel!r}")
        super().__init__(
            n_channels=n_channels, latent_dim=latent_dim, t_lat=t_lat,
            hidden_ch=hidden_ch, dilations=dilations, graph_mode=graph_mode,
            graph_k=graph_k, a_hat=a_hat,
            graph_out_mix_alpha=graph_out_mix_alpha,
            graph_kernel=graph_kernel, coords=coords,
            mexhat_gamma=mexhat_gamma, w_init=w_init,
            mexhat_gamma_m=mexhat_gamma_m, dog_init=dog_init)
        # 原出口逐通道投影删除（防死参数/死键：state_dict 无 graph_out.*；
        # 其 8,256 参数位由 exit_out 等量替换，见模块 docstring 参数记账）
        del self.graph_out
        # K 轮出口消息层（默认 PyTorch 初始化；K 默认 = graph_k = 2，与入口
        # 消息层数一致——design §2 表 K 行）
        self.exit_msg_layers = nn.ModuleList(
            [nn.Linear(hidden_ch, hidden_ch) for _ in range(graph_k)])
        # 零初始化出口投影（保初始 v ≡ 0 恒等起步——design §2 出口投影行）
        self.exit_out = nn.Linear(hidden_ch, latent_dim)
        nn.init.zeros_(self.exit_out.weight)
        nn.init.zeros_(self.exit_out.bias)

    def _forward_graph(self, z_t, t, z_obs, B, C, T_lat, D):
        """覆写尾段（design §1.1）：前段与包内现行逻辑逐行一致，出口换
        K 轮共享 A_eff 消息传递 + 零初始化投影。"""
        # 时间嵌入 broadcast 到 (B, C, T_lat, D)——同包内现行
        t_emb = self.time_proj(self.time_embed(t))   # (B, D)
        t_emb = t_emb.view(B, 1, 1, D).expand(B, C, T_lat, D)

        # 入口图混合器：per-channel proj + K 轮消息传递
        # → (B, C, T_lat, hidden)——同包内现行
        h = self.graph_mixer(z_t, z_obs, t_emb)

        # 膨胀卷积块：C 折叠进 batch（严格逐通道时间卷积；1x1 mix 只混
        # hidden 特征轴，无跨通道混合）——同包内现行
        x = h.permute(0, 1, 3, 2).reshape(B * C, self.hidden_ch, T_lat)
        for conv, norm1, act, mix, norm2, _ in [
                (b[0], b[1], b[2], b[3], b[4], b[5]) for b in self.blocks]:
            residual = x
            x = conv(x)
            x = act(norm1(x))
            x = act(norm2(mix(x)))
            x = x + residual

        x = self.out_norm(x)                          # (B*C, hidden, T_lat)
        # (B*C, hidden, T_lat) -> (B, C, T_lat, hidden)
        x = x.permute(0, 2, 1).reshape(B, C, T_lat, self.hidden_ch)

        # ── 出口图混合块（本变体唯一结构性差异，design §1.1）──────────
        # A_eff 与入口共享（σ/W/γ 单份、两应用位、梯度双路累计；用户拍板）。
        A = self.graph_mixer.effective_adjacency()    # (C, C) 有向
        for layer in self.exit_msg_layers:
            # 通道轴消息传递，与入口 LearnableMexHatMixer.forward 同式：
            # x_perm (B, T_lat, H, C) @ A.t() → agg 回排 (B, C, T_lat, H)
            x_perm = x.permute(0, 2, 3, 1)           # (B, T_lat, H, C)
            agg = x_perm @ A.t()                     # (B, T_lat, H, C)
            agg = agg.permute(0, 3, 1, 2)            # (B, C, T_lat, H)
            x = x + F.gelu(layer(agg))               # 逐轮残差，同入口形式
        # 零初始化出口投影 → v (B, C, T_lat, D)：初始严格 v ≡ 0
        return self.exit_out(x)


# =============================================================================
# CPU 快速构建自验（design §5 S1「初始 v≡0 自验」；内存内随机张量，
# 不读数据、不训练——静态口径，正式冒烟属 S4/S5）
# =============================================================================

def _self_check() -> int:
    torch.manual_seed(42)
    coords = torch.randn(128, 3)
    net = FlowVelocityNetExitGNN(graph_mode=True, graph_kernel="mexhat",
                                 coords=coords)
    z_t = torch.randn(2, 128, 16, 64)
    z_obs = torch.randn(2, 128, 16, 64)
    t = torch.rand(2)
    with torch.no_grad():
        v = net(z_t, t, z_obs)
    # ① 初始 v ≡ 0（exit_out 零初始化门控；恒等起步语义）
    assert v.shape == (2, 128, 16, 64), f"v shape {tuple(v.shape)}"
    n_nonzero = int(torch.count_nonzero(v).item())
    assert n_nonzero == 0, f"初始 v 非 0（{n_nonzero} 个非零元素）——exit_out 零初始化被破坏"
    # ② state_dict 键面：graph_out 删除不残留，exit_* 两族键在
    keys = net.state_dict()
    assert not any(k.startswith("graph_out") for k in keys), \
        "state_dict 残留 graph_out 键（del 未生效）"
    assert any(k.startswith("exit_msg_layers") for k in keys), \
        "state_dict 缺 exit_msg_layers 键"
    assert any(k.startswith("exit_out") for k in keys), "state_dict 缺 exit_out 键"
    # ③ 参数量：同参基线 + 33,024（exit_out 8,256 替换 graph_out 8,256 净 0，
    #    exit_msg_layers +33,024；A_eff 共享 +0——design §2 记账）
    base = FlowVelocityNet(graph_mode=True, graph_kernel="mexhat", coords=coords)
    n_base = sum(p.numel() for p in base.parameters())
    n_exit = sum(p.numel() for p in net.parameters())
    assert n_exit - n_base == 33024, \
        f"参数净增 {n_exit - n_base} != 预期 33024（design §2）"
    # ④ 反向一步冒烟：梯度经 exit_out 门控传导至消息层与共享 A_eff 参数
    v = net(z_t, t, z_obs)
    v.pow(2).mean().backward()
    assert net.exit_msg_layers[0].weight.grad is not None, \
        "exit_msg_layers 无梯度（梯度未穿过零初始化门控）"
    assert net.graph_mixer.W.grad is not None, "共享 A_eff 的 W 无梯度"
    n_sd = len(keys)
    print(f"[self-check] v≡0 OK（shape {tuple(v.shape)}，全零）")
    print(f"[self-check] state_dict OK：graph_out 键消失，exit_msg_layers/"
          f"exit_out 键在（共 {n_sd} 键）")
    print(f"[self-check] params：base {n_base:,} -> exit-gnn {n_exit:,}"
          f"（净 +{n_exit - n_base:,}，预期 +33,024）")
    print(f"[self-check] backward OK：exit_msg_layers 与共享 graph_mixer.W "
          f"均有梯度")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
