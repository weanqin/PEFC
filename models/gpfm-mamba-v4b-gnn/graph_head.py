"""GraphHead — latent-flow-v1 `LearnableMexHatMixer` 的适配移植（v4bg 唯一结构变化，design §2.2）。

源：`models/latent-flow-v1/latent_flow/models/oc_latent_flow.py::LearnableMexHatMixer`（约 313–520 行，
全程只读）。适配范围与等价性主张（design §2.3）：
- **数值等价（AC2 单测）**：`A_eff = γ_m·M(d;σ) + γ·W` 的构造——同 coords/σ/γ/W 下与源
  `effective_adjacency()` allclose rtol=1e-6（ricker + dog 两核）；距离核公式/参数化
  （log_sigma、dog 4 参数重参数化）/buffer 惯例（gamma_buf、gamma_m、pairwise_d）逐项对齐。
- **结构等价（布局适配）**：消息传递同构——源 `x_perm @ A.t()`（A[i,j]=j 驱动 i，行=靶/列=源），
  本实现 (B,T,C,h) 布局下等价 einsum `"ij,btjh->btih"`；K×{A_eff 聚合 + per-channel
  Linear(h→h)+GELU} 残差结构同源。
- **非等价（语境不同，如实登记）**：输入投影源为 Linear(2D→h)+t_proj（z_t/z_obs/t_emb 语境），
  本适配为 per-channel Linear(d_ch→h)（trunk 逐通道特征，时间条件已含在 18 维输入特征内）；
  另增 per-channel 输出投影 Linear(h→1)（v 读出，design §2.2 伪码）。
- 正则方法 `uniformity_penalty()` / `sparsity_penalty()` / `dog_params()` 保留源语义；
  **默认不进损失**（λ_u=λ_s=0，对齐 10-19 edra_fmt run 实际口径），旋钮在 config 暴露
  （`graph_regularization` 组合器；λ=0 返回精确 0 张量，逐位不改变基线损失）。

方向约定（design §2.2 / A-1）：本模块内部保持源语义 `A_eff[i,j] = 通道 j 驱动通道 i`（行=靶、
列=源）；全仓库 EC/GC 口径为行=源、列=靶——**一切对外读出统一转置**，转置发生在
`graph_readout.py` 单点（S4），本模块不转置（保等价测）。

几何单一 choke point（design §2.5/§4）：构造内加载 `geometry_edra_fmt.pt`（只读，62ch 10-20
坐标 + channel_names），并对数据根全 46 例 npz 断言 `channel_names` 顺序全等（A-7 升级；只读
npz header，lru_cache 按 (geometry, data_root) 缓存，重复构建零重扫）。v1 data.py 只读，
断言不落 dataset 侧。
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_GEOMETRY_PATH = "/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/edra_process_v4"
DEFAULT_EXPECT_N = 46  # edra status_ok 口径受试者数（HC24 + MDD22）


def _scan_npz_channel_names(data_root: str) -> List[Tuple[str, List[str]]]:
    """只读扫描数据根全部 npz 的 channel_names（HC/ + MDD/ 布局，edra 契约）。"""
    out: List[Tuple[str, List[str]]] = []
    if not os.path.isdir(data_root):
        raise ValueError(f"数据根不存在: {data_root}")
    for group in sorted(os.listdir(data_root)):
        gdir = os.path.join(data_root, group)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(gdir, fn)
            with np.load(path) as z:  # NpzFile 惰性读取：仅解压 channel_names 一个成员
                names = [str(x) for x in z["channel_names"].tolist()]
            out.append((path, names))
    return out


@lru_cache(maxsize=8)
def _geometry_gate(
    geometry_path: str, data_root: str, expect_n: Optional[int]
) -> Tuple[torch.Tensor, List[str], int]:
    """几何加载 + 通道对齐断言（单一 choke point 的实现；结果按参数缓存）。

    返回 (coords (62,3) float32, channel_names, 扫描 npz 数)。断言：
    - geometry 文件含 coords (C,3) 与 channel_names（长度一致）；
    - 数据根全部 npz 的 channel_names 与 geometry 顺序全等（零 mismatch 即刻报错）；
    - expect_n 非 None 时扫描数须恰等（默认 46 = edra status_ok 口径）。
    """
    g = torch.load(geometry_path, map_location="cpu", weights_only=False)
    coords = g["coords"].float()
    names = [str(x) for x in g["channel_names"]]
    if coords.dim() != 2 or coords.shape[1] != 3 or coords.shape[0] != len(names):
        raise ValueError(
            f"geometry 文件结构异常: coords {tuple(coords.shape)} vs {len(names)} names（应 (C,3)/(C,)）"
        )
    scanned = _scan_npz_channel_names(data_root)
    mismatch = [p for p, ns in scanned if ns != names]
    if mismatch:
        raise ValueError(
            f"通道名顺序与 geometry 不全等（{len(mismatch)}/{len(scanned)} 例 mismatch，"
            f"geometry={geometry_path}）：{mismatch[:3]}{'...' if len(mismatch) > 3 else ''}"
        )
    if expect_n is not None and len(scanned) != expect_n:
        raise ValueError(f"数据根 npz 数 {len(scanned)} != 期望 {expect_n}（data_root={data_root}）")
    return coords, names, len(scanned)


class GraphHead(nn.Module):
    """GNN 先验图头（LearnableMexHatMixer 适配版；接口 (B, T, C, d_ch) -> (B, T, C)）。

    结构（design §2.2 伪码）：
        x  = per-channel Linear(d_ch -> h)          # 跨通道权重共享，无混合
        for k in K:                                  # K 轮消息传递（默认 2）
            agg = A_eff @_C x                        # (…, C, h) 通道轴聚合（源语义 A[i,j]=j 驱动 i）
            x  = x + GELU(per-channel Linear(h -> h)(agg))
        v  = per-channel Linear(h -> 1)(x)           # (B, T, C)
        A_eff = γ_m · M(d; σ) + γ · W                # (C, C) 有向（W 不对称化）

    - M：距离核，`ricker`（1 参数 σ 可学习，零点钉在 d=σ）默认 / `dog`（4 参数激励/抑制宽度解耦）备选；
    - W：C×C 可学习残差，**零初始化**（`w_init` 张量可注入），不对称化 ⇒ A_eff 有向；
    - γ/γ_m 为 buffer（可 checkpoint 恢复）；pairwise_d 距离阵 register_buffer（源惯例）。
    """

    def __init__(
        self,
        n_channels: int = 62,
        d_ch: int = 1,
        hidden: int = 64,
        k_layers: int = 2,
        gamma: float = 0.1,
        gamma_m: float = 1.0,
        kernel: str = "ricker",
        w_init: Optional[torch.Tensor] = None,
        dog_init: Optional[dict] = None,
        geometry_path: str = DEFAULT_GEOMETRY_PATH,
        data_root: str = DEFAULT_DATA_ROOT,
        expect_n_subjects: Optional[int] = DEFAULT_EXPECT_N,
    ):
        super().__init__()
        # ---- 几何单一 choke point（构造期断言全例通道名顺序全等；不消耗 RNG）----
        coords, names, n_scanned = _geometry_gate(geometry_path, data_root, expect_n_subjects)
        if coords.shape[0] != n_channels:
            raise ValueError(
                f"geometry 通道数 {coords.shape[0]} != n_channels {n_channels}（{geometry_path}）"
            )
        self.channel_names = names  # 只读登记（非 tensor，不进 state_dict）
        self.n_scanned = n_scanned

        self.n_channels = n_channels
        self.d_ch = d_ch
        self.hidden_ch = hidden
        self.k_layers = k_layers
        self.gamma = float(gamma)
        assert kernel in ("ricker", "dog"), f"unknown kernel: {kernel!r}"
        self.kernel = kernel
        # γ（W 权重）与 γ_m（距离核权重）存 buffer（checkpoint 可恢复——A_eff 重建所需，源惯例）
        self.register_buffer("gamma_buf", torch.tensor(float(gamma)), persistent=True)
        self.register_buffer("gamma_m", torch.tensor(float(gamma_m)), persistent=True)

        # per-channel 输入/输出投影（跨通道权重共享，无混合；本适配的语境差异项，见模块 docstring）
        self.in_proj = nn.Linear(d_ch, hidden)
        self.out_proj = nn.Linear(hidden, 1)
        # K 轮消息传递的 per-channel 变换（共享权重）
        self.msg_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(k_layers)]
        )

        # 固定 pairwise 距离阵（buffer：随模块搬移 + checkpoint；两核共用，源惯例）
        d = torch.cdist(coords, coords)
        median_d = float(d[d > 0].median().item())
        self.register_buffer("pairwise_d", d.float(), persistent=True)

        # ---- 距离核参数（逐式对齐源实现）----
        if kernel == "ricker":
            # Ricker：1 个可学习参数 σ（零点钉在 d=σ），log_sigma 参数化
            self.log_sigma = nn.Parameter(torch.tensor(float(median_d)).log())
        else:
            # DoG：M = Ae·Ge − Ai·Gi，约束经重参数化恒成立（σ_i>σ_e、Ae>Ai），无 clamp/惩罚
            dog_init = dog_init or {}
            sigma_e = float(dog_init.get("sigma_e", median_d / 2.0))
            sigma_i = float(dog_init.get("sigma_i", median_d))
            Ae = float(dog_init.get("Ae", 1.0))
            Ai = float(dog_init.get("Ai", 0.6))
            assert sigma_i > sigma_e > 0, (
                f"DoG init requires sigma_i({sigma_i}) > sigma_e({sigma_e}) > 0"
            )
            assert Ae > Ai, f"DoG init requires Ae({Ae}) > Ai({Ai})"
            self.log_sigma_e = nn.Parameter(torch.tensor(float(sigma_e)).log())
            self.log_delta_sigma = nn.Parameter(
                torch.tensor(float(sigma_i - sigma_e)).log()
            )
            self.log_Ae = nn.Parameter(torch.tensor(float(Ae)).log())
            delta_a = float(Ae - Ai)
            self.log_delta_a = nn.Parameter(
                torch.tensor(float(math.log(math.expm1(delta_a))))
            )

        # 可学习残差 W（C×C）：默认零初始化（从纯带通先验出发学偏离）；w_init 注入给定拓扑；
        # 不对称化 ⇒ A_eff 有向（源语义）。零初始化不消耗 RNG。
        if w_init is not None:
            assert w_init.shape == (n_channels, n_channels), (
                f"w_init shape {tuple(w_init.shape)} != ({n_channels},{n_channels})"
            )
            self.W = nn.Parameter(w_init.clone().float())
        else:
            self.W = nn.Parameter(torch.zeros(n_channels, n_channels))

    # ---------------- A_eff（与源数值等价，AC2 断言） ----------------

    def effective_adjacency(self) -> torch.Tensor:
        """当前 A_eff = γ_m·M(d) + γ·W（(C,C) 有向，源语义 A[i,j] = 通道 j 驱动 i）。

        M 为 `self.kernel` 选择的对称距离核；γ_m 缩放核（0 = 消融先验）；W 不对称化。
        对外读出（矩阵导出/GC 对照）须在 graph_readout 单点转置为行=源/列=靶。
        """
        M = self._distance_kernel()
        return self.gamma_m * M + self.gamma_buf * self.W

    def _distance_kernel(self) -> torch.Tensor:
        """对称距离核 M(d)（逐式对齐源 `_distance_kernel`）。

        - ricker：`(1 − d²/σ²)·exp(−d²/(2σ²))`；
        - dog：`Ae·exp(−d²/(2σ_e²)) − Ai·exp(−d²/(2σ_i²))`（约束由重参数化恒成立）。
        """
        d2 = self.pairwise_d ** 2
        if self.kernel == "ricker":
            sigma = self.log_sigma.exp()
            s2 = sigma ** 2
            return (1.0 - d2 / s2) * torch.exp(-d2 / (2.0 * s2))
        sigma_e = self.log_sigma_e.exp()
        sigma_i = sigma_e + self.log_delta_sigma.exp()  # > sigma_e 恒成立
        Ae = self.log_Ae.exp()
        Ai = Ae - F.softplus(self.log_delta_a)          # < Ae 恒成立
        return (
            Ae * torch.exp(-d2 / (2.0 * sigma_e ** 2))
            - Ai * torch.exp(-d2 / (2.0 * sigma_i ** 2))
        )

    def sigma_value(self) -> float:
        """当前 σ*（ricker 核；dog 核返回 None——用 dog_params()）。"""
        if self.kernel != "ricker":
            return None  # type: ignore[return-value]
        return float(self.log_sigma.detach().exp())

    def dog_params(self) -> Dict[str, float]:
        """重建的 DoG 参数 {sigma_e, sigma_i, Ae, Ai}（dog 核；ricker 返回 {}）。（源同款）"""
        if self.kernel != "dog":
            return {}
        sigma_e = float(self.log_sigma_e.detach().exp())
        sigma_i = sigma_e + float(self.log_delta_sigma.detach().exp())
        Ae = float(self.log_Ae.detach().exp())
        Ai = Ae - float(F.softplus(self.log_delta_a.detach()))
        return {"sigma_e": sigma_e, "sigma_i": sigma_i, "Ae": Ae, "Ai": Ai}

    # ---------------- 正则（保留源语义；默认 λ=0 不进损失） ----------------

    def uniformity_penalty(self) -> torch.Tensor:
        """反均匀化正则：A_eff 行和方差过小（各通道均匀平均 = 各向同性化失败模式）时大。

        返回 1/(1+var(rowsum))（行=靶轴，与源同口径；+eps 稳定）。训练侧 λ_u × 本项加进损失。
        """
        A = self.effective_adjacency()
        rowsum = A.sum(dim=1)
        var = rowsum.var(unbiased=False) + 1e-6
        return 1.0 / (1.0 + var)

    def sparsity_penalty(self) -> torch.Tensor:
        """A_eff 的 L1 稀疏正则（仅非对角）：mean(|A_eff_ij|), i != j。（源同款）"""
        A = self.effective_adjacency()
        mask = ~torch.eye(self.n_channels, dtype=torch.bool, device=A.device)
        return A.abs()[mask].mean()

    # ---------------- 前向（消息传递；位置逐点，时间轴无交互） ----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, d_ch) -> v: (B, T, C)。

        聚合实现 `"ij,btjh->btih"` 与源 `x_perm @ A.t()`（(B,T,H,C) 布局）逐元素同构：
        agg[b,t,i,:] = Σ_j A_eff[i,j]·x[b,t,j,:]（源语义：j 驱动 i）。
        """
        x = self.in_proj(x)                              # (B, T, C, h) per-channel 投影
        A = self.effective_adjacency()                   # (C, C)
        for layer in self.msg_layers:
            agg = torch.einsum("ij,btjh->btih", A, x)    # A_eff @_C x（通道轴聚合）
            x = x + F.gelu(layer(agg))                   # 残差 + per-channel 变换
        return self.out_proj(x).squeeze(-1)              # (B, T, C)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def graph_regularization(
    head: GraphHead, lambda_uniform: float = 0.0, lambda_sparse: float = 0.0
) -> torch.Tensor:
    """λ_u·uniformity + λ_s·sparsity 组合器（config 旋钮 → 惩罚项的唯一装配点）。

    λ_u=λ_s=0（默认口径）返回**精确 0 标量**——`total + 0.0` 对正常浮点逐位不变（λ=0 时
    损失与基线逐位一致，单测断言）。λ>0 时返回 λ_u·head.uniformity_penalty() +
    λ_s·head.sparsity_penalty()（默认不启用；启用属口径变更须另登记）。
    """
    ref = next(head.parameters())
    if float(lambda_uniform) == 0.0 and float(lambda_sparse) == 0.0:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    total = torch.zeros((), device=ref.device, dtype=ref.dtype)
    if float(lambda_uniform) != 0.0:
        total = total + lambda_uniform * head.uniformity_penalty()
    if float(lambda_sparse) != 0.0:
        total = total + lambda_sparse * head.sparsity_penalty()
    return total
