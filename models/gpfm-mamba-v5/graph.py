"""graph.py — v5 空间块算子（A_eff 先验矩阵，本目录自含实现）。

**公式源 = `models/latent-flow-v1/latent_flow/models/oc_latent_flow.py::LearnableMexHatMixer`
（约 313–520 行，只读参照，机制逐式移植）**。按用户 2026-10-19 拍板（design §2.2）：
本文件与 `models/gpfm-mamba-v4b-gnn/graph_head.py` **无代码复用关系**（v4b 线实现停用不复用）。

机制（与源一致的要素）：
- ``A_eff = γ_m·M(d;σ) + γ·W``，(C, C) 有向（W 不对称化）；源语义 **A[i,j] = 通道 j 驱动 i
  （行=靶/列=源）**；对外可解释读出统一转置为行=源/列=靶（转置职责在读出侧单点承担）。
- ricker（默认，σ 可学习 ``log_sigma`` 参数化，过零点钉在 d=σ）/ dog（4 参数重参数化：
  σ_i = σ_e + exp(log_delta_sigma) > σ_e、A_i = A_e − softplus(log_delta_a) < A_e 恒成立，
  无 clamp/惩罚）双核。
- γ（W 权重）/ γ_m（距离核尺度）/ pairwise_d 为 buffer（checkpoint 可恢复——从存档重建 A_eff）。
- W (C, C) 零初始化（训练自纯几何先验出发学偏离）、不对称化（有向）、支持 w_init 注入。
- ``uniformity_penalty`` / ``sparsity_penalty`` / ``dog_params`` 源语义保留；
  λ_u/λ_s 默认 0 不进损失（口径对齐 10-19 edra_fmt run 实际口径——PRD R1「旋钮落地即可」）。

几何单一 choke point（A-7 处置沿用）：
- 构造内加载 ``geometry_edra_fmt.pt``（62ch 10-20 npy 基座，D2/D4 拍板产物，只读）；
- 数据根（旧范式 edra v4）**全 46 例 npz 的 channel_names 与几何名顺序全等断言**——
  几何行序与数据通道轴的唯一对齐防线（lru_cache：进程内每 (geometry, data_root) 只扫一次）。
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 几何/数据默认路径（config.geometry_path 透传覆盖；data_root 用 config.data_root 同值默认）
DEFAULT_GEOMETRY_PATH = "/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/edra_process_v4"
EXPECTED_GEOMETRY_SUBJECTS = 46  # edra v4 旧范式全量（HC 24 + MDD 22；status_ok 口径同数）


@lru_cache(maxsize=8)
def load_geometry(
    geometry_path: str = DEFAULT_GEOMETRY_PATH,
    data_root: str = DEFAULT_DATA_ROOT,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """几何单一 choke point（只读）：加载坐标 + 全量通道名对齐断言。

    返回 (coords (62,3) float32, channel_names 62 元组)。断言两道门禁（顺序固定，报错信息可区分）：
    1. 数据根每例 npz 的 ``channel_names`` 与几何文件名序**逐项全等**（顺序不一致即报错，
       报错含首个差异位）；
    2. 扫描例数 == 46（edra v4 全量口径；例数不符即报错）。
    lru_cache 按 (geometry_path, data_root) 缓存——同进程构造多模型只扫一次。
    """
    g = torch.load(geometry_path, map_location="cpu", weights_only=False)
    if not isinstance(g, dict) or "coords" not in g or "channel_names" not in g:
        raise ValueError(f"几何文件 {geometry_path} 缺 coords/channel_names 键（got keys: {list(g) if isinstance(g, dict) else type(g)}）")
    coords = torch.as_tensor(g["coords"], dtype=torch.float32)
    names = [str(x) for x in g["channel_names"]]
    if coords.shape != (62, 3) or len(names) != 62:
        raise ValueError(f"几何文件 {geometry_path} 形状异常：coords {tuple(coords.shape)} / names {len(names)}（应 (62,3)/62）")
    npz_files = sorted(Path(data_root).glob("*/*.npz"))
    if not npz_files:
        raise ValueError(f"几何门禁：数据根 {data_root} 未扫描到任何 npz（HC/ + MDD/ 布局）")
    for fp in npz_files:
        with np.load(fp) as z:
            got = [str(x) for x in z["channel_names"]]
        if got != names:
            first_diff = next(k for k in range(62) if got[k] != names[k])
            raise ValueError(
                f"几何门禁：{fp} channel_names 与几何名序不一致（位 {first_diff}: npz {got[first_diff]!r} != geometry {names[first_diff]!r}）"
            )
    if len(npz_files) != EXPECTED_GEOMETRY_SUBJECTS:
        raise ValueError(
            f"几何门禁：数据根 npz 例数 {len(npz_files)} != {EXPECTED_GEOMETRY_SUBJECTS}（edra v4 全量口径）"
        )
    return coords, tuple(names)


class MexHatAdjacency(nn.Module):
    """A_eff = γ_m·M(d;σ) + γ·W 空间算子（v5 trunk 跨块共享单实例）。

    参数（机制源 = latent-flow LearnableMexHatMixer，只读参照）：
    - n_channels: 通道数（=d_model，62）；
    - coords: 显式坐标 (62,3)（等价测/独立构造用）；None = 经几何 choke point 加载
      （geometry_path + data_root 全 46 例通道名断言）；
    - kernel: "ricker"（默认）| "dog"；
    - gamma / gamma_m: W 权重 / 距离核尺度（buffer，checkpoint 可恢复）；
    - w_init: None = W 零初始化；tensor (62,62) = 给定有向拓扑起步；
    - dog_init: dog 核初值 dict {sigma_e, sigma_i, Ae, Ai}（源默认：median_d/2 / median_d / 1.0 / 0.6）。
    """

    def __init__(
        self,
        n_channels: int = 62,
        coords: Optional[torch.Tensor] = None,
        kernel: str = "ricker",
        gamma: float = 0.1,
        gamma_m: float = 1.0,
        w_init: Optional[torch.Tensor] = None,
        dog_init: Optional[Dict] = None,
        geometry_path: str = DEFAULT_GEOMETRY_PATH,
        data_root: str = DEFAULT_DATA_ROOT,
    ):
        super().__init__()
        self.n_channels = n_channels
        # kernel 选择距离核形状："ricker"（1 参数 Mexican hat，默认）或 "dog"（4 参数
        # Difference-of-Gaussians——激励/抑制宽度解耦，σ 不再单独控制平衡）。
        assert kernel in ("ricker", "dog"), f"unknown kernel: {kernel!r}"
        self.kernel = kernel
        # γ/γ_m buffer 化（源同款）：checkpoint 可恢复，下游装载可重建 A_eff。
        self.register_buffer("gamma_buf", torch.tensor(float(gamma)), persistent=True)
        # γ_m 缩放（对称）距离核 M；默认 1.0 = 基线不变；0 = 消融几何先验臂。
        self.register_buffer("gamma_m", torch.tensor(float(gamma_m)), persistent=True)

        # 成对距离矩阵（buffer——随模块迁移设备且入 checkpoint；两核共享）。
        if coords is None:
            coords, _names = load_geometry(geometry_path, data_root)  # 几何 choke point（含 46 例断言）
        else:
            coords = torch.as_tensor(coords, dtype=torch.float32)
        d = torch.cdist(coords.float(), coords.float())
        median_d = float(d[d > 0].median().item())
        self.register_buffer("pairwise_d", d.float(), persistent=True)

        # ---- 距离核参数（逐核）----
        if kernel == "ricker":
            # Ricker：1 个可学习参数 σ（log_sigma 参数化；过零点钉在 d=σ）。
            self.log_sigma = nn.Parameter(torch.tensor(float(median_d)).log())
        else:
            # DoG: M = Ae·exp(−d²/2σe²) − Ai·exp(−d²/2σi²)，约束（σi>σe、Ae>Ai）经
            # 重参数化恒成立（无 clamp/惩罚）：
            #   σe = exp(log_sigma_e); σi = σe + exp(log_delta_sigma); Ae = exp(log_Ae);
            #   Ai = Ae − softplus(log_delta_a)
            dog_init = dog_init or {}
            sigma_e = float(dog_init.get("sigma_e", median_d / 2.0))
            sigma_i = float(dog_init.get("sigma_i", median_d))
            Ae = float(dog_init.get("Ae", 1.0))
            Ai = float(dog_init.get("Ai", 0.6))
            assert sigma_i > sigma_e > 0, (f"DoG init requires sigma_i({sigma_i}) > sigma_e({sigma_e}) > 0")
            assert Ae > Ai, f"DoG init requires Ae({Ae}) > Ai({Ai})"
            self.log_sigma_e = nn.Parameter(torch.tensor(float(sigma_e)).log())
            self.log_delta_sigma = nn.Parameter(torch.tensor(float(sigma_i - sigma_e)).log())
            self.log_Ae = nn.Parameter(torch.tensor(float(Ae)).log())
            # 解 softplus(x) = Ae − Ai → x = log(expm1(Ae−Ai))
            delta_a = float(Ae - Ai)
            self.log_delta_a = nn.Parameter(torch.tensor(float(math.log(math.expm1(delta_a)))))

        # 可学习残差 W（C×C）：零初始化（自纯带通几何先验出发学偏离）；w_init 提供则自给定
        # 拓扑起步。effective_adjacency 不对称化 ⇒ A_eff 有向（与初始化无关）。
        if w_init is not None:
            assert tuple(w_init.shape) == (n_channels, n_channels), (
                f"w_init shape {tuple(w_init.shape)} != ({n_channels},{n_channels})")
            self.W = nn.Parameter(w_init.clone().float())
        else:
            self.W = nn.Parameter(torch.zeros(n_channels, n_channels))

    def effective_adjacency(self) -> torch.Tensor:
        """当前 A_eff = γ_m·M(d) + γ·W（(C,C) 有向；M 对称、W 不对称化 ⇒ A_eff 可不对称）。

        方向语义（源同款）：A[i,j] = 通道 j 驱动 i（行=靶/列=源）——消息传递聚合式
        agg[..., i] = Σ_j A[i,j]·x[..., j]。
        """
        M = self._distance_kernel()
        A = self.gamma_m * M + self.gamma_buf * self.W
        return A

    def _distance_kernel(self) -> torch.Tensor:
        """对称距离核 M(d)（(C,C)）：ricker = (1−d²/σ²)·exp(−d²/2σ²)；dog = Ae·Ge − Ai·Gi。"""
        d2 = self.pairwise_d ** 2
        if self.kernel == "ricker":
            sigma = self.log_sigma.exp()
            s2 = sigma ** 2
            return (1.0 - d2 / s2) * torch.exp(-d2 / (2.0 * s2))
        # DoG
        sigma_e = self.log_sigma_e.exp()
        sigma_i = sigma_e + self.log_delta_sigma.exp()      # > σe 恒成立
        Ae = self.log_Ae.exp()
        Ai = Ae - F.softplus(self.log_delta_a)              # < Ae 恒成立
        return (Ae * torch.exp(-d2 / (2.0 * sigma_e ** 2))
                - Ai * torch.exp(-d2 / (2.0 * sigma_i ** 2)))

    def dog_params(self) -> Dict:
        """重建的 DoG 参数 {sigma_e, sigma_i, Ae, Ai}（读出/登记用）；ricker 返回 {}。"""
        if self.kernel != "dog":
            return {}
        sigma_e = float(self.log_sigma_e.detach().exp())
        sigma_i = sigma_e + float(self.log_delta_sigma.detach().exp())
        Ae = float(self.log_Ae.detach().exp())
        Ai = Ae - float(F.softplus(self.log_delta_a.detach()))
        return {"sigma_e": sigma_e, "sigma_i": sigma_i, "Ae": Ae, "Ai": Ai}

    def uniformity_penalty(self) -> torch.Tensor:
        """anti-uniformity 正则（源语义）：1/(1+rowsum.var()+ε)——A_eff 行和趋同（各向同性
        平均化失效模式）时大、行和分散（结构保留）时小；训练侧以 λ_u 加权进损失。"""
        A = self.effective_adjacency()
        rowsum = A.sum(dim=1)
        var = rowsum.var(unbiased=False) + 1e-6
        return 1.0 / (1.0 + var)

    def sparsity_penalty(self) -> torch.Tensor:
        """L1 稀疏正则（源语义）：A_eff 非对角 |·| 均值——驱动弱连接到精确零。"""
        A = self.effective_adjacency()
        mask = ~torch.eye(self.n_channels, dtype=torch.bool, device=A.device)
        return A.abs()[mask].mean()


def graph_regularization(
    adj: MexHatAdjacency, lambda_uniform: float, lambda_sparse: float
) -> torch.Tensor:
    """λ_u·uniformity + λ_s·sparsity 组合器（config 双 λ 旋钮的唯一消费点）。

    λ_u=λ_s=0（默认口径）时返回**精确 0 张量**——加进损失逐位不变（λ=0 不进损失的
    机器可读实现；启用属口径变更须另登记）。
    """
    reg = torch.zeros((), device=adj.W.device)
    if lambda_uniform != 0.0:
        reg = reg + lambda_uniform * adj.uniformity_penalty()
    if lambda_sparse != 0.0:
        reg = reg + lambda_sparse * adj.sparsity_penalty()
    return reg


class GraphSpatialStep(nn.Module):
    """空间块（design §2.1 逐字）：h + GELU(msg_i(A_eff @_axis h))。

    - msg_i = nn.Linear(d_model, d_model)（每块独立实例）；
    - A_eff 由 forward 形参传入（**跨块共享单实例**由模型持有——本模块不注册 adjacency
      子模块，避免共享参数在 state_dict 出现多路径副本）；
    - 聚合式（通道轴、逐时间位独立）：agg[b,l,i] = Σ_j A_eff[i,j]·h[b,l,j]
      （einsum "ij,blj->bli"；A[i,j]=j 驱动 i 源语义同 MexHatAdjacency）。
    """

    def __init__(self, d_model: int = 62):
        super().__init__()
        self.msg = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, A_eff: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,blj->bli", A_eff, x)
        return x + F.gelu(self.msg(agg))
